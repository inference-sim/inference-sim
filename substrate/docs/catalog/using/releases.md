# Releases and pinning

The catalog is released with the rest of blis-substrate: a git tag `vMAJOR.MINOR.PATCH` on
`main` (design §14.1), with a [GitHub release](https://github.com/inference-sim/blis-substrate/releases)
whose notes say what changed. A release embeds its catalog, so a simulator built on that
release reads exactly those files unless a run names another catalog directory.

## What a version promises

A PATCH release changes no contract, price or value. A new model, chip or field, or a changed
figure, is a MINOR release whose notes list what changed. Every Result records the catalog's
source and content hash, so a figure can be traced to the data behind it.

To see what moved between two releases:

```sh
git diff v0.11.0 v0.12.0 --stat -- catalog/
```

## Why a path must match the release

The loaders parse the catalog strictly. An unknown key is an error, not a field to ignore,
because an ignored field reads as zero, and a zero peak rate or token count gives a plausible
but wrong result with no warning. A simulator therefore reads the catalog formats of the
substrate release it was built with. A directory named with `--catalog` must be in those
formats: a catalog from a newer release that adds or renames a key is refused at load, with the
file and the key.

## Which docs version to read

The version menu at the top of this site has one entry per `MAJOR.MINOR` line, and each entry
describes the newest patch release on that line. `latest` is the highest release. `dev`
describes `main` and changes with every merge. Pre-releases are not published.

This page describes [[catalog.release]].
