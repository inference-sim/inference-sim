## Summary

<!-- One to three sentences: what this pull request does and why. -->

Closes #

## Change class

<!-- One of the two, by what the change does (DEV-12). A change that is both is split. -->

- [ ] Verification: the model's behaviour must not change (a refactor, a speed-up, or a fix
      that makes the code match an oracle or a written rule).
- [ ] Validation: it changes what is modelled (a pricing form, a coefficient, an input model).

## Invariants

<!-- The design invariants this change keeps or introduces, by identifier (for example SUB-6).
CI fails if an identifier does not exist. -->

## Tests

<!-- Commands run and their results. For each behaviour change: the test that fails without
it (X13). -->

```sh
```

## Evidence

<!-- Required for a validation change, or a verification change that moves a published
number: the price-change report and the pre-registered criteria (X13, VAL-6). -->

- [ ] Not needed: no published number moves.

## Data and names

- [ ] No third-party data committed; derived values cite their source by URL and commit or hash (X12).
- [ ] No private names (X12).
