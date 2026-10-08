# External / fork PR review (`/pr-review`)

External contributors cannot use the [L1 automated delivery loop](automated-delivery.md) — it is
gated to maintainer-authored work (#1813), because an AI agent that reads a PR and holds credentials
can be prompt-injected by a hostile author. `/pr-review` closes that gap: a maintainer runs the same
three reviews on **any** PR, including forks, on a footing where a successful injection is harmless.

## How to run it

Comment **`/pr-review`** on the pull request. Only a user with `admin`/`write`/`maintain` on the
repository triggers it; a stranger's comment does nothing. It posts one **advisory** comment with
three sections: architecture (archon), correctness (blis-pr-review), and a cross-vendor second
opinion (qa-review).

The verdict **gates nothing and merges nothing** — it is input for a human, not a status check.

## The threat model

The PR's code, diff, title, and comments are **attacker-controlled**. An LLM that reads them may be
instructed — through text hidden in those bytes — to exfiltrate a secret, run a command, or post
attacker content. We do not try to *prevent* injection (no filter is reliable); we **contain** it so
a successful injection can steal nothing and change nothing. This is the pattern established by
GitHub's own guidance and by projects that run LLM review on untrusted PRs (e.g.
`pytorch/pytorch`'s hardened PR review); it is the lesson of the 2026 "Comment and Control" finding,
where a PR *title* drove AI review actions into posting their own API keys as PR comments.

## The one invariant

> **PR code is READ, never EXECUTED** — and the LiteLLM key is never in the reviewer's session.

How each reviewer gets the PR without executing it:

| Reviewer | Needs | How, safely |
|---|---|---|
| **archon** (no LLM) | base & head Go trees | `git fetch pull/N/head` into the **object store only** (no checkout); the trusted-branch archon binary reads both trees. Static analysis — no `go build` on PR code. |
| **qa-review** (LLM) | read the PR files | a **safe read-only checkout** + `answerer.py --no-exec` (its code-executing `go` tool is dropped; `read_file`/`grep`/`list_dir` stay sandboxed to the checkout) |
| **blis-pr-review** (LLM) | read the PR files | the same checkout + a **tool-restricted** session: `Read`/`Grep`/`Glob` only, **no `Bash`/`Edit`/`WebFetch`/`WebSearch`/`Agent`**, writes confined to a verdict file by a `PreToolUse` hook |

"Safe read-only checkout" means: git hooks disabled (`core.hooksPath=/dev/null`),
`persist-credentials: false` (no token written where a Read tool could reach it), submodules off, and
escaping symlinks scrubbed (`scripts/pr_review/scrub_symlinks.sh`) before any reviewer reads the tree.

> blis runs the **blis-pr-review methodology read-only** — the real review perspectives (correctness,
> INV-* invariants, run/replay/observe parity, preemption/timeout, boundaries, behavioural test
> quality, R1–R23, docs) inlined into the prompt — rather than the stock `pr-review-toolkit` plugin.
> The plugin shells out (`Bash`/`gh`), which is both unsafe on untrusted code and non-functional
> without a shell, so it is not used on forks — the same reason `pytorch/pytorch` encodes its review
> as a read-only skill instead of a generic toolkit. **Read-only sub-agent fan-out** (`Agent`, to run
> the perspectives in parallel like the full toolkit) is a follow-up gated on the dry-run verifying
> that sub-agents inherit the no-`Bash` deny on the pinned action version (as pytorch verified).

## The containment, control by control

- **Maintainer-only trigger** (`.github/workflows/pr-review.yml`, `gate` job): the comment must match
  `/pr-review` as a precise token (not `/blis-pr-review`, `/archon-pr-review`, or a future
  `/pr-review-*`) **and** the commenter must hold write access. Both are checked from the trusted ref.
- **Three jobs so the box that reads untrusted code cannot act:** `gate` (ubuntu-latest, touches no PR
  content) → `review` (`pr-review-untrusted` runner, `contents: read`, **no** `pull-requests: write`,
  **no** `id-token: write`) → `post` (ubuntu-latest, `pull-requests: write`, never reads PR code).
  blis's `claude-code-action` is handed the job's read-only `github_token` so it takes its
  `OVERRIDE_GITHUB_TOKEN` path and **skips the OIDC exchange** — which would otherwise need
  `id-token: write` on the untrusted box and would mint the Anthropic App's *write* token (#1883).
- **Key out of the session:** LiteLLM needs the VPN, so the runner is self-hosted — but the key lives
  **only in a separate `litellm-proxy` pod** (`k8s/pr-review-runner.yaml`). The reviewer talks to the
  `litellm-proxy` Service with a **dummy** key; the proxy injects the real one (both
  `Authorization: Bearer` for qa and `x-api-key` for blis). A runtime step asserts no real key is in
  the job env. (It is a separate pod, not a same-pod sidecar, so the runner's egress can be locked
  tighter than the proxy's — see below.)
- **Egress lock — enabled, vanilla `NetworkPolicy`, no cluster-admin.** The runner pod runs under a
  default-deny egress policy that allows only DNS, the `litellm-proxy` pod (by label), and public `:443`
  with every private/VPC/metadata CIDR excluded. So the runner **cannot** reach LiteLLM, the metadata
  endpoint, or any in-cluster service directly; LiteLLM is reachable only *through* the proxy, which
  pins the hostname in nginx (VPC-LB IP rotation never touches the runner's policy). The earlier
  "vanilla policy breaks DNS" finding was a misdiagnosis: `172.21.0.10` is a ClusterIP, and Calico
  evaluates egress post-DNAT against the real CoreDNS pod, which a `namespaceSelector` peer on ports
  53/5353 matches — so DNS survives. Calico/`AdminNetworkPolicy`/cluster-admin are **not** required
  (that was the open ask in #1881; this closes it).
- **Output scrub:** the combined comment passes `scripts/pr_review/scrub_secrets.py` before posting —
  a last line, not the control (the control is that there is no key to leak).
- **Advisory:** the verdict is never a required status check, so an injected review cannot block a
  merge by failing the job.

## Why no `pull_request_target` two-stage split

`pytorch/pytorch` needs two workflows because its trigger (`pull_request_target`) hands a privileged
token into the untrusted PR context. `/pr-review` uses a **maintainer comment** instead, and no
reviewer holds a shell on untrusted input, so a single workflow with a write-access gate is enough.

## Infrastructure prerequisites (not created by the workflow)

- A self-hosted runner labelled **`pr-review-untrusted`** plus the **`litellm-proxy`** pod/Service and
  the three egress/ingress `NetworkPolicy` objects (`k8s/pr-review-runner.yaml`), isolated from the
  delivery-loop `self-hosted` pool.
- The proxy's `nous-wiki-llm` secret (LiteLLM endpoint + key). The workflow is inert-but-safe until
  these exist — it is maintainer-gated, so it cannot fire accidentally.

## Residual risks we accept

- **A weak/odd advisory comment.** An injection could nudge the LLM to write something unhelpful. It
  is visible, scrubbed, and non-authoritative; a human reads it.
- **Gateway spend.** Bounded by the LiteLLM budget on the shared key; the session's dummy key is useless
  (it only reaches the `litellm-proxy`, which overwrites it).
- **Proxy reachable namespace-wide.** The proxy listens on a routable pod IP; the namespace-wide
  `allow-same-namespace` policy lets any blis pod reach `:4000` (an `litellm-proxy-ingress` policy
  records the runner-only intent for if that blanket policy is tightened). The credential cannot be
  read back through the proxy — it is injected outbound only — so the exposure is use as a
  namespace-local LiteLLM relay, which any blis pod could already obtain from `nous-wiki-llm` directly.
- **Platform trust.** We trust `claude-code-action` and the runner image to hold; actions are
  SHA-pinnable.

## Validation before enabling

Because the verdict is advisory, green CI is not the bar. Before relying on this: run an **adversarial
dry-run** on a throwaway fork PR carrying an injection payload (title, comment, and a file body) and
confirm the key is not exfiltrated, no PR code executes, and the comment is harmless; capture the
containment proof (no real key in the job env); and get a human maintainer sign-off.
