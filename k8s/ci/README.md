# CI runner scale sets

Two ARC (`gha-runner-scale-set`) scale sets that run CI's slowest jobs on the cluster instead of on
GitHub-hosted runners. They run pull request code, so they hold no credentials and reach nothing in the cluster
(Isolation, below).

| Scale set | Pod | Max pods | Jobs |
|---|---|---|---|
| `blis-ci-large` | 16 CPU, 32 GiB (limit 48 GiB) | 3 | ci.yml: `substrate kernel`, `cmd cluster matrix`, `vllm-race (kernelmodel-census)` |
| `blis-ci-oracle` | 4 CPU, 16 GiB | 6 | oracle-vllm.yml: scenarios, inertness, 2 fuzz jobs; oracle-sglang.yml: 4 `oracle-run` jobs |

`ci.yml`'s `changes` job decides once where the heavy jobs run, and every job reads its outputs
(`runner-large`, `runner-oracle`, and `large-workers` for the worker counts only the larger pods can
afford, `BLIS_PERTURBATION_WORKERS` and `BLIS_CLUSTER_MATRIX_WORKERS`):

```yaml
SAME_REPO: ${{ github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository }}
```

A job goes to a scale set only when its variable is set and the run is this repository's own code:
not a pull request, or one whose head is in this repository. A fork's pull request, or one whose
head repository was deleted (`head.repo` is then null), runs hosted. With the variables unset,
everything runs hosted as before. The gate jobs (`changes`, `required`), `byte gate arm64`,
`release.yml` and the fast jobs stay hosted. No required check changes name.

This routes jobs; it does not protect the cluster. A fork's pull request runs its own copy of the
workflows, so it can name a scale set's label in `runs-on`. What protects the cluster is the
isolation below. So before approving a fork's workflow run, check its diff under `.github/` for a
changed `runs-on` or `vars.` reference. A repository-level scale set cannot be limited to selected
workflows; an organization-level runner group can, if the scale sets move to the organization.

## Isolation

- **Namespace.** `blis` holds the CI scale sets and nothing else (`namespace.yaml`,
  `network.yaml`). The quota, the network policies and the EgressFirewall cover every pod in it.
  The agent runners (`k8s/deployment.yaml`, `k8s/pr-review-runner.yaml`), which hold credentials,
  move to a namespace of their own first (setup step 1): in `blis` they would lose the API server
  and every host off the list.
- **Nodes.** The cluster does not let us label or taint nodes, so the pods schedule on any worker
  node, beside other namespaces' pods. What a container escape would need is closed off instead:
  the pods are unprivileged (below), mount no host path, and hold nothing worth taking.
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
  the pod's, so they are not on the list. The rules name hosts, some by wildcard
  (`*.blob.core.windows.net`, `*.actions.githubusercontent.com`), and several are CDN-fronted with
  addresses that rotate. So setup step 6 confirms that the cluster's OpenShift version takes
  wildcard `dnsName` rules and that no fetch is refused over a run. If either fails, an egress proxy
  with a domain allow-list, which the pods are only allowed to reach, replaces the EgressFirewall.
- **Quota.** `namespace.yaml` caps requests at 72 CPU and 200 GiB, the most both sets can start.

## Setup

1. **Quota and network.** The `blis` namespace already exists. Move the agent runners out of it
   first: the policies below cover every pod in `blis`.

   ```bash
   kubectl apply -f k8s/ci/namespace.yaml -f k8s/ci/network.yaml
   ```

2. **GitHub credential.** A GitHub App installed on this repository only, with *Administration:
   read and write* (runner registration) and nothing else (`github_app_id`,
   `github_app_installation_id`, `github_app_private_key`), or a fine-grained token with that one
   permission. It is a secret in `blis` that only the ARC controller reads, from its own namespace:

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

5. **Enable.** First make `required` a required check of the `main` ruleset (an admin's change): it
   is the job that waits on every other, so a failing oracle or a job on the cluster blocks a merge
   only through it. Then set the repository variables:

   ```bash
   gh variable set CI_RUNS_ON_LARGE --body blis-ci-large
   gh variable set CI_RUNS_ON_ORACLE --body blis-ci-oracle
   ```

6. **Confirm the egress list.** For the first run, set the EgressFirewall's last rule to `Allow` and
   turn on egress logging. Run CI on a pull request that turns on every group (a change to
   `.github/workflows/ci.yml` does), add any host the log shows that the list lacks, then restore the
   `Deny` and run it again.

## While the cluster is down

A job queued for a scale set with no runner waits, up to 24 hours, and `required` waits with it, so
pull requests cannot merge. Delete the variable, then re-run all jobs: the `changes` job decides
again and the jobs run hosted.

## Rollback

Delete a variable (`gh variable delete CI_RUNS_ON_LARGE`) and its jobs run hosted again from the next
run. Nothing in the workflows needs to change.

## Cache

The repository's Actions cache holds 10 GB, and the oldest entries are evicted past it. A
self-hosted runner's oracle environment is a cache entry of its own (the key holds
`runner.environment`, since a venv links to its interpreter by path), so enabling `CI_RUNS_ON_ORACLE`
adds about 1.8 GB on `main` (vLLM about 1.0 GB, SGLang about 0.8 GB). The budget on `main`:

| Entry | Size |
|---|---|
| Go build cache | about 0.75 GB |
| Oracle environments, hosted and self-hosted | about 3.6 GB |
| SGLang source, fuzz corpora, tools | about 0.5 GB |

That is about 5 GB, which leaves the other half for open pull requests' own entries.
