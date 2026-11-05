# Releases and pinning

A *release* is a git tag of the form `MAJOR.MINOR.PATCH`, with no `v` prefix, together with a [GitHub release](https://github.com/inference-sim/blis-catalog/releases) whose notes say what changed. Clone a release, not `main`.

## Why pin

The programs that read the catalog parse it strictly. An unknown key is an error, not a field to ignore, because an ignored field reads as zero, and a zero peak rate or token count gives a plausible but wrong result with no warning. As a result, a release that adds, renames or restructures a key stops an older consumer at startup. Pinning a release prevents that, and it also means two people who run the same scenario months apart read the same data.

inference-sim declares the release it is tested against in the [Catalog compatibility](https://inference-sim.github.io/inference-sim/latest/getting-started/installation/#catalog-compatibility) section of its installation guide. blis-latency-kernel and blis-schemas take a catalog path as input and check whatever they are given against the schema version they were built with.

## Upgrading

Read the release notes, clone the new release beside the old one, and rerun your scenarios against it. Switch when they succeed. The pin records which release a tool was tested with; it does not prevent you from using another.

## Which docs version to read

The version menu at the top of this site has one entry per `MAJOR.MINOR` line, and each entry describes the newest patch release on that line. `latest` is the highest release. `dev` describes `main` and changes with every merge. Pre-releases are not published.

This page describes [[catalog.release]].
