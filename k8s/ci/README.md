# CI runner scale sets

Two ARC (`gha-runner-scale-set`) scale sets that run CI's slowest jobs on the cluster instead of on
GitHub-hosted runners. They run pull request code, so they hold no credentials and reach nothing in the cluster
(Isolation, below).

| Scale set | Pod | Max pods | Jobs |
|---|---|---|---|
| `blis-ci-large` | 16 CPU, 32 GiB (limit 48 GiB) | 3 | ci.yml: `substrate kernel`, `cmd cluster matrix`, `vllm-race (kernelmodel-census)` |
| `blis-ci-oracle` | 4 CPU, 16 GiB | 6 | oracle-vllm.yml: scenarios, inertness, 2 fuzz jobs (oracle-sglang.yml can use it, but ci.yml runs it hosted) |

`ci.yml`'s `changes` job decides once where the heavy jobs run, and every job reads its outputs
(`runner-large`, `runner-oracle`, and `large-workers` for the worker counts only the larger pods can
afford, `BLIS_PERTURBATION_WORKERS` and `BLIS_CLUSTER_MATRIX_WORKERS`).

A job goes to a scale set whenever its variable is set, whoever opened the pull request: a fork's
pull request runs where this repository's own branches run. With the variables unset, everything
runs hosted as before. The gate jobs (`changes`, `required`), `byte gate arm64`, `release.yml` and
the fast jobs stay hosted. No required check changes name.

What keeps a fork's code off the cluster until someone has read it is approval, not routing: the
repository's Actions setting *Require approval for all external contributors* (setup step 0) holds
every outside contributor's workflow run until a maintainer approves it. A fork's pull request runs
its own copy of the workflows, so it can name a scale set's label in `runs-on`. So before approving
a fork's run, read its diff, and under `.github/` look for a changed `runs-on` or `vars.` reference.
A repository-level scale set cannot be limited to selected workflows; an organization-level runner
group can, if the scale sets move to the organization.

## Isolation

- **Approval.** Every outside contributor's workflow run waits for a maintainer's approval before
  any job starts (setup step 0). This does most of the work: the rest limits what an approved run
  that turns out to be hostile can do.
- **Namespace.** `blis` holds the CI scale sets, their ARC controller and its listeners, and nothing
  else (`namespace.yaml`, `network.yaml`). The quota, the network policies and the EgressFirewall
  cover every pod in it.
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
- **No credentials.** A runner pod mounts no service account token and no secret except its own
  one-job registration. The GitHub credential and the controller's service account token live in
  the namespace, but only the controller and listener pods may reach the API server
  (`allow-arc-api-server`), and a pull request's code runs in neither: a runner pod cannot reach the
  API server to read them. The job's own `GITHUB_TOKEN`, which `actions/checkout` keeps in
  `.git/config`, is read-only: ci.yml and oracle-vllm.yml declare `permissions: {contents: read}`.
- **Egress.** `network.yaml` denies all ingress, and denies egress except DNS and ports 80/443 to
  public addresses. The EgressFirewall then narrows egress to the hosts CI fetches from: GitHub and
  its Actions endpoints, the Go proxy and checksum database, the Go and Node downloads, PyPI, the
  PyTorch CPU index, and Hugging Face (configuration files only). Image pulls are the node's, not
  the pod's, so they are not on the list. The rules name hosts, some by wildcard
  (`*.blob.core.windows.net`, `*.actions.githubusercontent.com`), and several are CDN-fronted with
  addresses that rotate. **Open:** this cluster's OVN-Kubernetes rejects the four wildcard
  `dnsName` rules (`kubectl -n blis get egressfirewall default` reports *EgressFirewall Rules not
  correctly applied*), so until an egress proxy with a domain allow-list, which the pods are only
  allowed to reach, replaces the EgressFirewall, egress is what `network.yaml` allows: any public
  address on ports 80 and 443.
- **Limits.** Every runner pod has CPU, memory and ephemeral-storage limits (`values-*.yaml`), and
  every job that can run on a scale set has a `timeout-minutes`, so one run cannot hold the cluster.
  `namespace.yaml` caps the namespace at 73 CPU and 200 GiB of requests, the most both sets can
  start plus the controller and listeners, whose requests and limits come from its LimitRange.

## Setup

0. **Approval.** Require approval for every outside contributor's workflow run (an admin's change;
   in the settings it is Actions, General, *Require approval for all external contributors*):

   ```bash
   gh api -X PUT repos/inference-sim/inference-sim/actions/permissions/fork-pr-contributor-approval \
     -f approval_policy=all_external_contributors
   ```

1. **Quota and network.** The `blis` namespace already exists. Move the agent runners out of it
   first: the policies below cover every pod in `blis`.

   ```bash
   kubectl apply -f k8s/ci/namespace.yaml -f k8s/ci/network.yaml
   ```

2. **GitHub credential.** A GitHub App installed on this repository only, with *Administration:
   read and write* (runner registration) and nothing else (`github_app_id`,
   `github_app_installation_id`, `github_app_private_key`), or a fine-grained token with that one
   permission. It is a secret in `blis` that only the ARC controller and its listeners read:

   ```bash
   kubectl -n blis create secret generic blis-ci-github --from-file=github_token=<token file>
   ```

3. **Controller.** The ARC controller (`gha-runner-scale-set-controller`) runs in `blis` and
   watches only `blis`, so the CI set-up shares no namespace with anything else. It and the
   listeners it starts call the API server, which `allow-arc-api-server` (`network.yaml`) allows
   for them alone; its addresses there are this cluster's (`kubectl get svc,endpoints kubernetes -n
   default`). The CRDs are cluster-wide, so a cluster that already has them skips them:

   ```bash
   helm install arc-blis -n blis --skip-crds --set flags.watchSingleNamespace=blis \
     oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller --version <chart>
   ```

4. **Scale sets.**

   ```bash
   for s in large oracle; do
     helm install blis-ci-$s -n blis -f k8s/ci/values-$s.yaml \
       --set controllerServiceAccount.name=arc-blis-gha-rs-controller \
       --set controllerServiceAccount.namespace=blis \
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
