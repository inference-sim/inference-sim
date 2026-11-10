# Releasing BLIS repositories

Every BLIS release is made by the release bot, `tools/release` in this repository, or, while
the bot is unavailable, by a person with write access to the repository, following the manual
procedure below (design §13.6, LIF-3). Both produce the same tags in the same order. blis-substrate's first release is
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
  unless the release fixes only that issue, is a security release, or the person cutting the
  release records a waiver in the release notes (X27).

## A review set

A change that spans repositories is one review set, described by a `stack.lock` file (the format
is in `tools/release/stack.go`). Stack CI builds the heads together through a `go.work` it
generates and never commits (X11). Once the set is approved, the bot releases it bottom-up
(X10): blis-substrate, then the engine repositories and the router fork, then inference-sim, then
the applications. For each repository the bot stages the final release commit on a branch
`release/<tag>`: it merges the head into the base branch and writes the pins to the published
tags of the layers below it in one commit. It tags nothing at that point. The commit it stages is not the head
that was checked, so a person pushes the branch and CI runs on that exact commit. The bot reads
the results from that commit's own check runs, through `.github/check-runs.yaml`, which names
the check runs that prove each registry check. Only then does the bot make the annotated tag,
whose message is the generated release notes (LIF-4), on that same commit, and only if every ci
check has a successful run there (LIF-3). A hand-written result cannot stand in for a run: there
is no results file. A tip that moved after the check has no check runs and is refused, and so is
a check whose job was skipped, which proves nothing. When a job ran more than once on the commit
(a re-run, or another manual run), its newest attempt decides: an earlier failure that a later
run of the same job replaced does not hold the tag back, and an earlier pass does not hide a later
failure. A newest attempt that is still running is no result. Every leg of a matrix is still
needed. A check that runs only nightly (the method-error job of VAL-5) is proved by running the
nightly workflow by hand on the staged commit (`workflow_dispatch`). A head that already carries a
release tag is refused, and so is a
`go.mod` that requires a BLIS module at a pseudo-version.

The bot works one layer at a time and a person does the pushing and the waiting. Each run re-plans the whole
`stack.lock` from the repositories as they stand, so it can be repeated after any interruption:
a head that is already staged keeps its tag and its branch, a head whose tag is already made is
reported as done and skipped, and the version numbers never advance. It is refused only when the
whole set is already released.

For each layer, lowest first:

1. `-apply` stages the layer (`-layer <repo>`, by default the first layer not yet tagged). A
   layer above the first is staged only when the tag of every layer below it exists and is
   published on that repository's origin, because its `go.mod` is pinned to those real tags.
2. The person pushes `release/<tag>` and runs the layer's checks on that exact commit. A manual
   run of the workflow (`workflow_dispatch`) runs every job, including those a path filter
   skips on a pull request.
3. `-tag` makes the annotated tag on that tip, if its check runs prove every ci check, and tags
   nothing else.
4. The person pushes the tag and runs the layer's checks against it. Then they run `-apply` for the
   next layer.

The command prints the next step after each run. The push to `main` needs a bypass for the
release bot or a person with write access, because the branch requires a pull request and an
approving review; the tag push needs the same, because tags are protected (LIF-1).

A dry run prints each tag, pin, refusal and the notes, and changes nothing:

```sh
go run ./tools/release -lock stack.lock \
    -dir blis-substrate=../blis-substrate -dir blis-vllm=../blis-vllm -dir inference-sim=../inference-sim
```

Pass `-dir` once per repository in the set. A set that holds an engine or a router also needs
`-dir inference-sim=<checkout>`, even when inference-sim is not released: the substrate bound of
design §3.2 reads inference-sim's `go.mod`, so the checkout is read-only input. The bot refuses
the plan up front when it is missing, is not inference-sim's module, or does not hold exactly one
direct `blis-substrate` requirement.

`-apply` stages one layer's release commit on `release/<tag>` in the local checkout and tags
nothing. After CI has run on the staged commit, `-tag` reads its check runs (`-check-runs` names the
mapping, `.github/check-runs.yaml` by default) and tags that exact commit. `-sign` signs the pin commits and tags with the key git is configured with;
without it they are annotated and unsigned. The command never pushes.

## The manual procedure

Any person with write access to the repository can run it, with standard git or the hosting
service's interface. It needs no second person and no signing key. Another person can read each
step against the bot's dry run before it is taken, and the project may ask for one; the procedure
does not.

1. Run the bot's dry run on the approved `stack.lock`. If the bot cannot run at all, work through
   the list under "What the bot checks" by hand and write down each result.
2. If anything is refused, stop. The refusal is fixed in a pull request, or, for X27 only, a
   waiver is written into the release notes.
3. For each repository in bottom-up order:
   1. Merge the head into the base branch with a merge commit (`git merge --no-ff <head>`), with
      the head commit stated in the message.
   2. If the dry run lists pins, set exactly those versions in `go.mod` (`go get <module>@<tag>`
      for each), run `go mod tidy`, and commit with the subject `Pin <module> <tag>, ... for <tag>`.
   3. Create the tag the dry run named: `git tag -a <tag> -m <notes> <commit>`, where the notes are
      the dry run's, ending with a final line `Tagged by the manual procedure: <person>` (LIF-3).
      Signing the tag is encouraged and not required (X20).
   4. Append `<tag> <commit>` to `.github/tags.lock` in the same pull request as the next
      repository's pin, or in a follow-up pull request for the top repository (LIF-1).
   5. Push the branch, then the tag: `git push origin <tag>`. Wait for the next layer's checks
      against the new tag to pass.
4. Nobody moves or deletes a tag. A mistake gets a new tag (LIF-1).

A nightly check confirms that every tag made by a person is annotated and listed in its release
notes (LIF-3), and that every tag still resolves to the commit `.github/tags.lock` records
(LIF-1).
