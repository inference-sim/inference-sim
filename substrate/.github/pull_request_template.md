<!-- BEGIN GENERATED: invariants (go run ./tools/invariants render; do not edit) -->
## Summary

<!-- One to three sentences: what this pull request does and why. -->

## Change class

<!-- Exactly one, by what the change does (DEV-12, design §8.9). A change that is both is split. -->

- [ ] Verification: the model's behaviour must not change (a refactor, a speed-up, or a fix that
      makes the code match an oracle or a written rule).
- [ ] Validation: it changes what is modelled (a pricing form, a coefficient, an input model).

## Invariants

<!-- The identifiers this change keeps or introduces, one per line, such as "- SUB-6". CI fails
if an identifier does not exist (X14). The list of identifiers is in AGENTS.md. -->

## Closes

<!-- One "Closes <owner>/<repo>#<n>" line per issue. -->

## Tests

<!-- Commands run and their results. For each behaviour change, the test that fails without
it (X13). -->

## Evidence

<!-- Required for a validation change, and for a verification change that moves a published
number: the success criteria committed in advance, the held-out evidence, the price-change
report and the scoring harness's figures on published units (X13, VAL-6). Otherwise write
"Not needed: no published number moves." -->

## Depends on

<!-- Other pull requests, by URL and commit. "None" if none. -->

## Not in this pull request

<!-- What remains, and why. -->

## Review questions

<!-- CI adds a required checklist item here for each review question whose paths this pull
request changes (design §13.2). A code owner answers each one: tick it, and write "yes" or
"no" with a one-line reason. -->

## Checklist

- [ ] X12: No third-party data committed; each derived value cites its source by URL and commit or hash; no private names.
- [ ] X13: Each behaviour change has a test that fails without it.
- [ ] X14: The commit hashes listed in this description are the branch heads.
- [ ] X20: Each commit written with an agent's help has one Assisted-by line and no Co-authored-by line naming a tool; the person responsible adds the DCO sign-off.
- [ ] X24: Every review finding is fixed, filed as a linked issue, or rejected with a reason.
- [ ] X26: Every path, symbol, issue and supported-feature statement in changed documentation is generated or checked.
<!-- END GENERATED: invariants -->
