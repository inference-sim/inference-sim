# CI runner scale sets

Two ARC (`gha-runner-scale-set`) scale sets that run CI's slowest jobs on the cluster instead of on
GitHub-hosted runners. They run pull request code, so they are kept apart from the agent runners
in `k8s/deployment.yaml` and `k8s/pr-review-runner.yaml`, which hold credentials.

| Scale set | Pod | Max pods | Jobs |
|---|---|---|---|
| `blis-ci-large` | 16 CPU, 32 GiB (limit 48 GiB) | 3 | ci.yml: `substrate kernel`, `cmd cluster matrix`, `vllm-race (kernelmodel-census)` |
| `blis-ci-oracle` | 4 CPU, 16 GiB | 6 | oracle-vllm.yml: scenarios, inertness, 2 fuzz jobs; oracle-sglang.yml: 3 `oracle-run` jobs |

Each job reads `runs-on` from a repository variable and falls back to `ubuntu-latest`:

```yaml
runs-on: ${{ !github.event.pull_request.head.repo.fork && vars.CI_RUNS_ON_ORACLE || 'ubuntu-latest' }}
```

So with the variables unset, everything runs hosted as before. A pull request from a fork always
runs hosted. The worker counts that only the larger pods can afford (`BLIS_PERTURBATION_WORKERS`,
`BLIS_CLUSTER_MATRIX_WORKERS`) follow the same expression. The gate jobs (`changes`, `required`),
`byte gate arm64`, `release.yml` and the fast jobs stay hosted. No required check changes name.

## Isolation

- **Namespace and nodes.** Pods run in `blis-ci` (`namespace.yaml`), only on nodes labeled
  `blis.ci/pool=ci` and tainted `blis.ci/dedicated=ci:NoSchedule`, so they never share a node with
  the agent runners.
- **Ephemeral and unprivileged.** One job per pod, then the pod is deleted. uid 1001, no privilege
  escalation, all capabilities dropped, `RuntimeDefault` seccomp, no Docker daemon or container
  mode.
- **No credentials.** No service account token is mounted, and no secret is mounted except the
  runner's own one-job registration. The GitHub App secret that ARC reads lives in the namespace,
  but a pod cannot reach the API server to read it.
- **Egress.** `network.yaml` denies all ingress, and denies egress except DNS and ports 80/443 to
  public addresses. The EgressFirewall then narrows egress to the hosts CI fetches from: GitHub and
  its Actions endpoints, the Go proxy and checksum database, the Go and Node downloads, PyPI, the
  PyTorch CPU index, and Hugging Face (configuration files only).
- **Quota.** `namespace.yaml` caps requests at 72 CPU and 200 GiB, the most both sets can start.

## Setup

1. **Nodes.** Label and taint the CI worker pool:

   ```bash
   kubectl label node <node> blis.ci/pool=ci
   kubectl taint node <node> blis.ci/dedicated=ci:NoSchedule
   ```

2. **Namespace, quota and network.**

   ```bash
   kubectl apply -f k8s/ci/namespace.yaml -f k8s/ci/network.yaml
   ```

3. **Image.** Build `runner.Dockerfile` (GitHub's `actions-runner`, pinned by digest, plus `gcc`
   for `go test -race`), push it to the registry the cluster pulls from, and put the pushed
   reference `REGISTRY/blis-ci-runner@sha256:...` into both values files.

4. **GitHub App secret.** A GitHub App installed on the repository with *Administration: read and
   write* (runner registration) only:

   ```bash
   kubectl -n blis-ci create secret generic blis-ci-github-app \
     --from-literal=github_app_id=<id> --from-literal=github_app_installation_id=<id> \
     --from-file=github_app_private_key=<key.pem>
   ```

5. **Scale sets.** The ARC controller is installed once per cluster (`gha-runner-scale-set-controller`).
   Then:

   ```bash
   for s in large oracle; do
     helm install blis-ci-$s -n blis-ci -f k8s/ci/values-$s.yaml \
       oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set --version <chart>
   done
   ```

   On OpenShift, let the runner service accounts run as uid 1001:

   ```bash
   for s in large oracle; do
     oc adm policy add-scc-to-user nonroot-v2 -z blis-ci-$s-gha-rs-no-permission -n blis-ci
   done
   ```

6. **Enable.** Set the repository variables:

   ```bash
   gh variable set CI_RUNS_ON_LARGE --body blis-ci-large
   gh variable set CI_RUNS_ON_ORACLE --body blis-ci-oracle
   ```

7. **Confirm the egress list.** For the first run, set the EgressFirewall's last rule to `Allow` and
   turn on egress logging. Run CI on a pull request that turns on every group (a change to
   `.github/workflows/ci.yml` does), add any host the log shows that the list lacks, then restore the
   `Deny` and run it again.

## Rollback

Delete a variable (`gh variable delete CI_RUNS_ON_LARGE`) and its jobs run hosted again from the next
run. Nothing in the workflows needs to change.
