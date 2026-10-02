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

> blis runs as a **bounded single session**, not the full `pr-review-toolkit` orchestration — that
> needs `Bash`/`Agent`, which is unsafe on untrusted input. The external review trades the toolkit's
> depth for a locked-down tool surface.

## The containment, control by control

- **Maintainer-only trigger** (`.github/workflows/pr-review.yml`, `gate` job): the comment must match
  `/pr-review` as a precise token (not `/blis-pr-review`, `/archon-pr-review`, or a future
  `/pr-review-*`) **and** the commenter must hold write access. Both are checked from the trusted ref.
- **Three jobs so the box that reads untrusted code cannot act:** `gate` (ubuntu-latest, touches no PR
  content) → `review` (`pr-review-untrusted` runner, `contents: read`, **no** `pull-requests: write`)
  → `post` (ubuntu-latest, `pull-requests: write`, never reads PR code).
- **Key out of the session:** LiteLLM needs the VPN, so the runner is self-hosted — but the key lives
  **only in a sidecar container** (`k8s/pr-review-runner.yaml`). The reviewer talks to
  `http://localhost:4000` with a **dummy** key; the sidecar injects the real one (both
  `Authorization: Bearer` for qa and `x-api-key` for blis). A runtime step asserts no real key is in
  the job env.
- **Egress lock — follow-up (not yet enabled).** A default-deny egress `NetworkPolicy` is the intended
  belt-and-suspenders, but on this cluster the pod's DNS resolver (`172.21.0.10`) is a node-local/host
  resolver that neither a `namespaceSelector` nor an `ipBlock: 0.0.0.0/0` egress peer matches, so every
  vanilla `NetworkPolicy` form tried blocked DNS and broke the runner (a non-443 port *was* correctly
  blocked, so the policy enforces — it just can't thread cluster DNS). Enabling it needs an
  `AdminNetworkPolicy` or node-CIDR allowance, tracked as a follow-up. This is acceptable for v1 because
  the **primary** control is that no high-value secret is in the session: the LiteLLM key is in the
  sidecar, and the job's `GITHUB_TOKEN` is read-only (`contents`/`pull-requests: read`), so there is
  nothing worth exfiltrating even over open 443.
- **Output scrub:** the combined comment passes `scripts/pr_review/scrub_secrets.py` before posting —
  a last line, not the control (the control is that there is no key to leak).
- **Advisory:** the verdict is never a required status check, so an injected review cannot block a
  merge by failing the job.

## Why no `pull_request_target` two-stage split

`pytorch/pytorch` needs two workflows because its trigger (`pull_request_target`) hands a privileged
token into the untrusted PR context. `/pr-review` uses a **maintainer comment** instead, and no
reviewer holds a shell on untrusted input, so a single workflow with a write-access gate is enough.

## Infrastructure prerequisites (not created by the workflow)

- A self-hosted runner labelled **`pr-review-untrusted`** with the **LiteLLM sidecar**
  (`k8s/pr-review-runner.yaml`), isolated from the delivery-loop `self-hosted` pool.
- The sidecar's `nous-wiki-llm` secret (LiteLLM endpoint + key). The workflow is inert-but-safe until
  these exist — it is maintainer-gated, so it cannot fire accidentally.
- (Follow-up) the egress `NetworkPolicy` once the cluster-DNS issue above is resolved.

## Residual risks we accept

- **A weak/odd advisory comment.** An injection could nudge the LLM to write something unhelpful. It
  is visible, scrubbed, and non-authoritative; a human reads it.
- **Gateway spend.** Bounded by the LiteLLM budget on the shared key; the session's dummy key is useless
  (it only reaches the in-pod sidecar).
- **No egress-lock yet** (see the follow-up above). Mitigated by the keyless session + read-only token:
  there is no high-value secret on the box to exfiltrate.
- **Platform trust.** We trust `claude-code-action` and the runner image to hold; actions are
  SHA-pinnable.

## Validation before enabling

Because the verdict is advisory, green CI is not the bar. Before relying on this: run an **adversarial
dry-run** on a throwaway fork PR carrying an injection payload (title, comment, and a file body) and
confirm the key is not exfiltrated, no PR code executes, and the comment is harmless; capture the
containment proof (no real key in the job env); and get a human maintainer sign-off.
