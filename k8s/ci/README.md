# CI runner scale sets

Two ARC (`gha-runner-scale-set`) scale sets that run CI's slowest jobs on the cluster instead of on
GitHub-hosted runners. They run pull request code, so they hold no credentials and reach nothing in the cluster
(Isolation, below).

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

- **Namespace.** Pods run in `blis` (`namespace.yaml`, `network.yaml`). The quota, the network
  policies and the EgressFirewall cover every pod in the namespace, so the agent runners
  (`k8s/deployment.yaml`, `k8s/pr-review-runner.yaml`) must run in another namespace on a cluster
  that hosts these scale sets: in `blis` they would lose the API server and every host off the list.
  The pods schedule on any worker node; no node is labeled or tainted for them.
- **Ephemeral and unprivileged.** One job per pod, then the pod is deleted. uid 1001, no privilege
  escalation, all capabilities dropped, `RuntimeDefault` seccomp, no Docker daemon or container
  mode.
- **No image of our own.** An init container copies the runner out of GitHub's `actions-runner`
  image into the home volume, and the job runs in `buildpack-deps:noble`, which has the C toolchain
  `go test -race` needs (cgo). Both are public images pinned by digest; there is nothing to build
  or push.
- **No credentials.** No service account token is mounted, and no secret is mounted except the
  runner's own one-job registration. The GitHub credential that ARC reads lives in the namespace,
  but a pod cannot reach the API server to read it.
- **Egress.** `network.yaml` denies all ingress, and denies egress except DNS and ports 80/443 to
  public addresses. The EgressFirewall then narrows egress to the hosts CI fetches from: GitHub and
  its Actions endpoints, the Go proxy and checksum database, the Go and Node downloads, PyPI, the
  PyTorch CPU index, and Hugging Face (configuration files only). Image pulls are the node's, not
  the pod's, so they are not on the list.
- **Quota.** `namespace.yaml` caps requests at 72 CPU and 200 GiB, the most both sets can start.

## Setup

1. **Quota and network.** The `blis` namespace already exists:

   ```bash
   kubectl apply -f k8s/ci/namespace.yaml -f k8s/ci/network.yaml
   ```

2. **GitHub credential.** A fine-grained token with *Administration: read and write* (runner
   registration) on the repository only, or a GitHub App with that permission
   (`github_app_id`, `github_app_installation_id`, `github_app_private_key`):

   ```bash
   kubectl -n blis create secret generic blis-ci-github --from-file=github_token=<token file>
   ```

3. **Controller.** The ARC controller (`gha-runner-scale-set-controller`) watches `blis` from a
   namespace of its own: it and the listeners it starts call the API server, which `network.yaml`
   denies in `blis`. The CRDs are cluster-wide, so a cluster that already has them skips them:

   ```bash
   helm install arc-blis -n <controller namespace> --skip-crds --set flags.watchSingleNamespace=blis \
     oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller --version <chart>
   ```

4. **Scale sets.**

   ```bash
   for s in large oracle; do
     helm install blis-ci-$s -n blis -f k8s/ci/values-$s.yaml \
       --set controllerServiceAccount.name=arc-blis-gha-rs-controller \
       --set controllerServiceAccount.namespace=<controller namespace> \
       oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set --version <chart>
   done
   ```

   On OpenShift, let the runner service accounts run as uid 1001:

   ```bash
   for s in large oracle; do
     oc adm policy add-scc-to-user nonroot-v2 -z blis-ci-$s-gha-rs-no-permission -n blis
   done
   ```

5. **Enable.** Set the repository variables:

   ```bash
   gh variable set CI_RUNS_ON_LARGE --body blis-ci-large
   gh variable set CI_RUNS_ON_ORACLE --body blis-ci-oracle
   ```

6. **Confirm the egress list.** For the first run, set the EgressFirewall's last rule to `Allow` and
   turn on egress logging. Run CI on a pull request that turns on every group (a change to
   `.github/workflows/ci.yml` does), add any host the log shows that the list lacks, then restore the
   `Deny` and run it again.

## Rollback

Delete a variable (`gh variable delete CI_RUNS_ON_LARGE`) and its jobs run hosted again from the next
run. Nothing in the workflows needs to change.
