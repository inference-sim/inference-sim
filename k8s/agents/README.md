# Agent runner scale sets

Two ARC (`gha-runner-scale-set`) scale sets for the workflows that call an LLM, in the `blis`
namespace beside the CI scale sets (`k8s/ci/README.md`), driven by the same ARC controller. They
replace the runner Deployments that ran these jobs on the old cluster.

| Scale set | Pod | Max pods | Pod label | Jobs |
|---|---|---|---|---|
| `blis-ci-agents` | 2 CPU, 4 GiB (limit 8 GiB) | 6 | `blis-runner: agents` | `claude.yml` (`claude`, `claude-review`), `archon.yml` (`archon-review`), `deliver-implement.yml` (`deliver`), `deliver-verify.yml` (`verify`), `deliver-correct.yml` (`correct`) |
| `blis-ci-pr-review` | 2 CPU, 4 GiB (limit 8 GiB) | 2 | `blis-runner: pr-review` | `pr-review.yml` (`review`) |

Each job names its scale set in `runs-on` (`blis-ci-agents`, `blis-ci-pr-review`). The gate
jobs (`check-permissions`, `authorize`, `gate`, `post`, `report-status`) stay on hosted runners, so a comment from someone without write access never starts a pod.

The two sets hold different things, and the pods are kept apart by that label:

- **`blis-ci-agents`** runs trusted code only (this repository's default branch and the delivery
  branches). Its jobs hold `LITELLM_API_KEY` and `LITELLM_BASE_URL` in their step env and a
  `GITHUB_TOKEN` with write scopes. They call LiteLLM directly, so these pods alone may reach its
  addresses.
- **`blis-ci-pr-review`** reads an untrusted pull request and runs none of it. It holds the
  proxy's caller token (`LITELLM_PROXY_TOKEN`) and a read-only token, and reaches LiteLLM only
  through the `litellm-proxy` pod (`litellm-proxy.yaml`), which checks the caller token and adds
  the real key from the secret `blis-litellm`.

`scripts/agent_runners_test.go` checks the laws below on the manifests and workflows.

## Isolation

- **One label grants the extra paths.** `k8s/ci/network.yaml` covers every pod in `blis`: ingress
  denied, egress only to DNS and public addresses outside the intranet (`9.0.0.0/8`), no API
  server. NetworkPolicies add up, so `network.yaml` here opens one path per label: `9.0.0.0/8` on
  443 for `blis-runner: agents` and the proxy, and the proxy's port 4000 for
  `blis-runner: pr-review`. The label is set in the scale sets' values files, and no CI values file
  sets it. A job cannot change its own pod's labels: the pod has no service account token and no
  route to the API server. Which pool, and so which label, a job gets is its `runs-on`, though,
  and an approved fork run can name either pool (Risks, 5).
- **The proxy admits pr-review pods only.** Its ingress allows `blis-runner: pr-review` alone, so a
  CI pod (running a pull request's code) cannot use it as a relay. The old namespace allowed any pod
  in it to reach the proxy. It relays only the routes the reviewers call (Claude Code's
  `/v1/messages` and its `count_tokens`, qa-review's `/chat/completions`, the preflight's
  `/v1/models`), returns 404 for any other LiteLLM route, and sends the key only to an upstream whose
  certificate verifies for the upstream's name.
- **The proxy requires a caller token.** It relays a request only when `x-api-key` or a Bearer
  `Authorization` carries the token from the secret `blis-litellm-proxy-token`, and answers 401
  otherwise, without contacting LiteLLM. The repository secret `LITELLM_PROXY_TOKEN` holds the same
  value, and only `pr-review.yml`'s `review` job references it. A fork's run gets no repository
  secrets, so it cannot use the proxy even from a pr-review pod. The token is worth nothing outside
  a pr-review pod, and the comment scrubber redacts it.
- **No secret is readable from a runner.** `blis-litellm`, `blis-ci-github` and the controller's
  token live in the namespace, but a runner pod mounts no service account token and cannot reach
  the API server, so it cannot read them. The proxy reads `blis-litellm` and
  `blis-litellm-proxy-token` through its own env, and a CI pod cannot reach the proxy.
- **One label per pool.** A scale set's only label is its name, so an agent job can never land on
  the pr-review pool. (The old pr-review runner also carried `self-hosted`, so a trusted delivery
  job could be scheduled on it.)
- **Ephemeral and unprivileged.** As the CI pods: one job per pod, uid 1001, no privilege
  escalation, every capability dropped, `RuntimeDefault` seccomp, no Docker daemon, no image of our
  own (`buildpack-deps:noble`, with `gh` and `jq` from a checksum-pinned init container).

## Risks

These are accepted with the decision to run the agent pods in `blis` rather than a namespace of
their own (the discussion's plan, #2683).

1. **Shared namespace with pull request code.** CI pods run approved pull requests' code in the same
   namespace and on the same nodes as the pods that hold the LiteLLM key and write tokens. The
   network policies above keep a CI pod from the agent pods, the proxy and the secrets, so what is
   left is a container escape to the node, which every runner pod's restrictions guard against
   (unprivileged, no host path, `RuntimeDefault` seccomp). A namespace of its own would not change
   that, since the cluster does not let us dedicate nodes.
2. **One quota.** `k8s/ci/namespace.yaml`'s quota covers CI and agent pods together. It is sized
   for every scale set at its maximum at once, so a full CI load cannot keep an agent pod from
   starting.
3. **The intranet range on 443 for the agent pods.** LiteLLM's load balancer rotates its addresses
   inside `9.0.0.0/8`, so the agent pods may reach any address there on 443. The old runners had no
   egress policy at all.
4. **The EgressFirewall.** `k8s/ci/network.yaml`'s `EgressFirewall` is namespace-wide and not in
   force on this cluster (it rejects wildcard rules). If it ever applies, it must allow LiteLLM's
   addresses too, or the agent jobs and the proxy lose LiteLLM.
5. **A fork run can name an agent pool.** The label keeps a CI pod off these paths, but a job picks
   its pool by `runs-on`, and a fork's pull request runs its own copy of the workflows. So an
   approved fork run that turns out to be hostile can name `blis-ci-pr-review` and reach the proxy,
   which answers 401 without the caller token (a fork's run gets no repository secrets), and the
   public internet on 443 like any CI pod; or name `blis-ci-agents` and reach `9.0.0.0/8` on 443,
   with no secret either. This is
   not a regression: the old runners took any job that named `self-hosted`, with no egress policy
   at all. What guards it today is approval and the approver's check for a changed `runs-on`
   (`k8s/ci/README.md`). A repository-level scale set cannot be limited to selected workflows; the
   fix is organization-level runner groups for these two scale sets, each restricted to the
   workflows that use it at `refs/heads/main` (a follow-up for an organization admin).

## Setup

1. **LiteLLM credential.** A secret in `blis` that only the proxy reads. `base-url` is a bare
   `https://host`, no path and no trailing slash: the proxy appends the request path, and sends the
   host as `Host`.

   ```bash
   kubectl -n blis create secret generic blis-litellm \
     --from-literal=base-url=https://<host> --from-file=api-key=<key file>
   ```

   The agent jobs use the repository secrets `LITELLM_BASE_URL` and `LITELLM_API_KEY`; set them to
   the same credential (`gh secret set`).

   **Caller token.** The proxy relays only requests that carry it. Make one, store it as the
   secret `blis-litellm-proxy-token` and as the repository secret `LITELLM_PROXY_TOKEN`, then
   delete the file:

   ```bash
   openssl rand -hex 32 > <token file>
   kubectl -n blis create secret generic blis-litellm-proxy-token --from-file=token=<token file>
   gh secret set LITELLM_PROXY_TOKEN < <token file>
   ```

2. **Quota and network.** `k8s/ci/namespace.yaml` has room for these pods and the proxy's Service.

   ```bash
   kubectl apply -f k8s/ci/namespace.yaml -f k8s/agents/network.yaml -f k8s/agents/litellm-proxy.yaml
   ```

3. **Scale sets**, with the same controller and GitHub credential as the CI sets:

   ```bash
   for s in agents pr-review; do
     helm install blis-ci-$s -n blis -f k8s/agents/values-$s.yaml \
       --set controllerServiceAccount.name=arc-blis-gha-rs-controller \
       --set controllerServiceAccount.namespace=blis \
       oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set --version 0.15.1
     oc adm policy add-scc-to-user nonroot-v2 -z blis-ci-$s-gha-rs-no-permission -n blis
   done
   ```

4. **Check the proxy** from a pr-review pod's point of view, once one runs (or run `/pr-review` on a
   pull request: its *Preflight litellm-proxy* step asks the proxy for `/v1/models` without the
   caller token, which must get 401, and with it, which must get 200).

The jobs take these pools as soon as the workflows naming them are on the default branch.
