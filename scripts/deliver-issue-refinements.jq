# Renders the DESIGN REFINEMENTS an implement phase must honour out of a sub-issue's comment
# thread, from `{"comments": [...]}` — `gh issue view <N> --json comments` — with one extra
# field per comment, `writeAccess`, that the caller has resolved against the collaborator
# permission API. Emits a markdown digest on stdout, oldest comment first, and nothing at all
# when no comment qualifies.
#
# WHY THIS EXISTS (#1782). The implement phase used to read the issue BODY and nothing else. When a
# design was refined in the comment thread after the body was written — a narrowed scope, a
# corrected contract, an "actually do X, not Y" — the agent never saw it, faithfully implemented an
# out-of-date spec, and the divergence surfaced only in verify or in human review, an agent hour
# later. This filter is the channel that closes that gap.
#
# It lives in a file rather than inline in deliver-implement.yml so that it can be tested
# (scripts/deliver_issue_refinements_test.go). Both failure directions are expensive and neither is
# visible without reading a real delivery: selecting too little ignores a real correction, and
# selecting too much lets a comment from someone with no write access steer a delivery that runs on
# a self-hosted runner with credentials in its environment.
#
# ── The trust term ────────────────────────────────────────────────────────────────────────────
#
# `writeAccess` is supplied by the caller, NOT derived here, and a comment missing the field is
# dropped — the filter fails closed. It is the caller's job because establishing it needs a live
# API call (`repos.getCollaboratorPermissionLevel`, admin/write/maintain), which nothing in this
# repository can evaluate offline.
#
# `authorAssociation` is deliberately NOT used for this, and the reason is a measurement rather
# than a preference: on issue #1782 in this repository the maintainer who issues every
# `/approve-issue-for-pr-delivery` command reports `authorAssociation: CONTRIBUTOR`, because they
# have had a PR merged and GitHub reports that in preference to their collaborator status. Trusting
# OWNER/MEMBER/COLLABORATOR would therefore have dropped exactly the comments this feature exists
# to read, while still admitting anyone whose PR has ever been merged. The repository already has
# one trust boundary for this loop — the collaborator permission check the delivery command and the
# review triggers both use — so this reuses it instead of inventing a second, weaker one.
#
# ── What else is dropped, and why each matters ────────────────────────────────────────────────
#
#   - BOT authors. The delivery loop itself comments on the issue (blocked-dependency refusals,
#     tracking-issue refusals, no-work reports). Feeding its own prose back to the agent as a
#     design refinement is a loop, and a bot cannot hold a design opinion the body does not.
#   - A comment with no author (a deleted account) — there is nobody to attribute authority to.
#   - MINIMIZED comments. Hiding a comment as off-topic, spam or outdated is a human explicitly
#     saying it does not count; honouring it anyway would overrule that.
#   - Comments that are nothing but SLASH COMMANDS. `/approve-issue-for-pr-delivery` is on every
#     delivered issue by construction, so admitting it would mean every single delivery reports a
#     refinement and the count stops meaning anything. A comment that carries prose ALONGSIDE a
#     command is kept — the prose may well be the refinement.
#   - Empty and whitespace-only bodies (a reaction-only comment).
#
# ── Ordering ──────────────────────────────────────────────────────────────────────────────────
#
# Sorted by `createdAt` ascending with `id` as a tie-break, so the order is total and identical
# across runs. This is load-bearing rather than cosmetic: the authority rule the prompt states is
# "where two refinements conflict, the LATER one wins", which is only well defined if the digest
# has one fixed notion of later. The tie-break covers two comments sharing a timestamp, where
# `sort_by` alone would leave the winner up to the API's response order.

def is_bot:
  (.author.is_bot == true)
  or ((.author.login // "") | test("\\[bot\\]$"));

def has_author:
  ((.author.login // "") | test("[^[:space:]]"));

def has_prose:
  ((.body // "") | test("[^[:space:]]"));

# True when every non-blank line of the body is a slash command, so the comment carries no prose
# of its own. The command shape is `/word` optionally followed by arguments — narrow on purpose, so
# a line that merely begins with a path (`/tmp/foo is where it lands`) is prose and keeps the
# comment.
def slash_command_only:
  ((.body // "")
   | split("\n")
   | map(select(test("^[[:space:]]*/[A-Za-z][A-Za-z0-9_.-]*([[:space:]].*)?$") | not))
   | map(select(test("[^[:space:]]")))
   | length) == 0;

(.comments // . // [])
| (if type == "array" then . else [] end)
| map(select(type == "object"))
| map(select(.writeAccess == true))
| map(select(is_bot | not))
| map(select(has_author))
| map(select(.isMinimized != true))
| map(select(has_prose))
| map(select(slash_command_only | not))
| sort_by([(.createdAt // ""), (.id // .url // "")])
| to_entries
| map([
    "### Refinement \(.key + 1) — @\(.value.author.login) at \(.value.createdAt // "an unknown time")",
    (.value.url // ""),
    "",
    (.value.body // ""),
    ""
  ])
| flatten
| join("\n")
