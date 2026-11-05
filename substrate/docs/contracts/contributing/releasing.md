# Releasing BLIS repositories

Every BLIS release is made by the release bot, `tools/release` in this repository, or, while
the bot is unavailable, by two release owners following the manual procedure below (design
§13.6, LIF-3). Both produce the same tags in the same order. blis-substrate's first release is
tagged by the manual procedure, because the bot's code arrives with it (release
plan §3.1).

## What the bot checks

The bot refuses to tag a repository when any of these holds:

- An invariant check the repository owns, with status `ci` in the registry
  (`contracts/invariants/`), has no passing result at the commit being tagged (LIF-3).
- An invariant's target milestone has been reached and the invariant has not reached its target
  status (design §13.3). A milestone is reached by its milestone release; a release within the
  milestone is not held to its targets.
- An engine or router tag breaks LIF-5: it is not `v<upstream>-blis.N` (or the `-post.K.blis.N`
  and `-rev.W.blis.N` forms), `N` is not one above the previous `N` on the same upstream release,
  or it is a plain `v<upstream>`.
- An engine or router release requires a newer blis-substrate than inference-sim does (design
  §3.2). Move inference-sim first, in the same review set.
- A MAJOR release, or a MINOR release at v0, still contains a deprecated feature (LIF-7). The
  features release plan §3.3 relaxes until M4 are listed in `tools/release/relaxations.go`.
- An open issue labelled `invariant-bug` lives in the repository or in one whose code it ships,
  unless the release fixes only that issue, is a security release, or a release owner records a
  waiver in the release notes (X27).

## A review set

A change that spans repositories is one review set, described by a `stack.lock` file (the format
is in `tools/release/stack.go`). Stack CI builds the heads together through a `go.work` it
generates and never commits (X11). Once the set is approved, the bot releases it bottom-up
(X10): blis-substrate, then the engine repositories and the router fork, then inference-sim, then
the applications. For each repository it merges the head into the base branch, writes the pins
to the tags just made below it in one commit, and makes a signed, annotated tag whose message is
the generated release notes (LIF-4). The release job then pushes the branch and the tag, runs the
next layer's checks against the new tags, and only then releases the next repository.

A dry run prints each tag, pin, refusal and the notes, and changes nothing:

```sh
go run ./tools/release -lock stack.lock -results results.json \
    -dir blis-substrate=../blis-substrate -dir blis-vllm=../blis-vllm -dir inference-sim=../inference-sim
```

`-apply` makes the merges, pin commits and tags in the local checkouts; `-sign` signs them. The
command never pushes.

## The manual procedure

Two release owners (the `release-owners` group in each repository's CODEOWNERS) run it together.
One acts; the other checks each step against the bot's dry run before it is taken.

1. Both owners run the bot's dry run on the approved `stack.lock` and compare the output. If the
   bot cannot run at all, they work through the list under "What the bot checks" by hand and write
   down each result.
2. If anything is refused, stop. The refusal is fixed in a pull request, or, for X27 only, a
   waiver is written into the release notes and signed off by both owners.
3. For each repository in bottom-up order:
   1. Merge the head into the base branch with a merge commit (`git merge --no-ff <head>`), with
      the head commit stated in the message.
   2. If the dry run lists pins, set exactly those versions in `go.mod` (`go get <module>@<tag>`
      for each), run `go mod tidy`, and commit with the subject `Pin <module> <tag>, ... for <tag>`.
      The commit is signed and carries no sign-off, as a release-bot commit would (X20).
   3. Create the tag the dry run named, annotated and signed with the acting owner's key, with the
      dry run's notes as its message, and a final line `Tagged by the manual procedure: <owner>,
      checked by <owner>` (LIF-3).
   4. Append `<tag> <commit>` to `.github/tags.lock` in the same pull request as the next
      repository's pin, or in a follow-up pull request for the top repository (LIF-1).
   5. Push the branch, then the tag. Wait for the next layer's checks against the new tag to pass.
4. Neither owner moves or deletes a tag. A mistake gets a new tag (LIF-1).

A nightly check confirms that every tag made by a person is signed and listed in its release
notes (LIF-3), and that every tag still resolves to the commit `.github/tags.lock` records
(LIF-1).
