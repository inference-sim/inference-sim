# Releases and pinning

The registry is released with the rest of blis-substrate: a git tag `vMAJOR.MINOR.PATCH`
on `main` (design §14.1). A release fixes every value in every set, and embeds them, so a
simulator built on that release reads exactly those values unless a run names another
registry directory.

## What a version promises

A PATCH release changes no contract, price or value. Refitting a family, adding a part or
adding a set changes values, and is a MINOR release whose notes list every changed value
(design §14.1). Record the release next to any figure computed with it; every Result
records it, with the registry's content hash.

To see what moved between two releases:

```sh
git diff v0.11.0 v0.12.0 --stat -- registry/coefficients/
```

## Versions of this site

The documentation is published once per release, so the reference pages always show the
numbers of the version being read.

| In the version selector | Built from | Updated |
|---|---|---|
| `X.Y` | the newest release `vX.Y.Z` of that line | when a release is published |
| `latest` | the newest release line | moves to each new release line |
| `dev` | `main` | on every merge |

The site's root opens `latest`. Every other version shows a banner saying it is not the
newest release.
