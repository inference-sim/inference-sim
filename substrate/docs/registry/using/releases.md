# Releases and pinning

A release is a git tag, `vMAJOR.MINOR.PATCH`, on `main`, with a GitHub release that
describes what changed. It fixes every value in every set. A consumer that records which
release it read can reproduce its coefficients exactly.

## Pin the exact tag

The version number orders releases. It does not promise that values are unchanged
between them: refitting a family, adding a part and adding a set all change what a
scenario resolves to, and each is a normal reason to release, at any of the three
levels. Pin the full tag, and record it next to any figure computed with it.

```sh
git clone --branch <!-- registry:release --> --depth 1 https://github.com/inference-sim/blis-registry.git
```

To see what moved between two releases, compare them:

```sh
git diff v0.1.0 v0.1.1 --stat -- coefficients/
```

The [releases page](https://github.com/inference-sim/blis-registry/releases) summarizes
each one and links the pull requests behind it.

## Versions of this site

The documentation is published once per release, so the reference pages always show the
numbers of the version being read.

| In the version selector | Built from | Updated |
|---|---|---|
| `X.Y.Z` | the release tag `vX.Y.Z` | when the release is published |
| `latest` | the highest release | moves to each new highest release |
| `dev (main)` | `main` | on every merge |

The site's root opens `latest`. Every other version shows a banner saying it is not the
newest release. Versions are full release numbers, not `X.Y`, because a patch release can
change values, and each release's pages must show that release's values.
