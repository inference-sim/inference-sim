# Selects the delivery PRs a stall sweep should consider, from `gh pr list --state open
# --json number,headRefName,labels,createdAt`. Emits one `<number>\t<createdAt>` line per
# candidate, and nothing when there are none.
#
# It lives in a file rather than inline in deliver-stall-sweep.yml so that it can be tested
# (scripts/deliver_stall_candidates_test.go). Selecting the wrong PR here means labelling a
# healthy delivery `needs-human` and halting it, which is the sweep's worst failure mode, and
# "verified by hand over seven shapes once" is not a property that survives the next edit.
#
# A delivery is in flight when all four hold:
#
#   1. Its head ref is a delivery branch. Anchored `^...$` so a near-miss like
#      `deliver/issue-12-wip` is not swept — this loop names its branches exactly.
#   2. It is open. (The caller passes `--state open`; not re-checked here.)
#   3. It carries NEITHER terminal label. Either one means the loop already reached a verdict
#      and stopped, so silence is expected rather than suspicious.
#   4. It is NOT paused. This one is easy to omit and expensive to omit: a paused delivery goes
#      quiet BY DESIGN, so it crosses any quiet threshold every single time. Sweeping it would
#      apply `needs-human` over a human's explicit hold and force manual cleanup before the
#      delivery could resume — the automation overruling the operator, which is the opposite of
#      what the pause exists for.
.[]
| select(.headRefName | test("^deliver/issue-[0-9]+$"))
| select([.labels[].name] | any(. == "ready-for-merge" or . == "needs-human") | not)
| select([.labels[].name] | any(. == "deliver:paused") | not)
| "\(.number)\t\(.createdAt)"
