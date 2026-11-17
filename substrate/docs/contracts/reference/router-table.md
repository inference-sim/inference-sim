# Router table

<p class="lede">An entry of the router table maps one router BLIS does not host, in one mode,
to the policy of llm-d's router that stands closest to it, with the evidence for the match
(design §11.3 and §11.4).</p>

The entries are in the catalog, as `routers/<name>.yaml`, and the [catalog reference](../../catalog/reference/routers.md)
shows them. `LoadRouterEntry` reads one strictly, and `Entry.Validate` checks it. The name is the
file's stem and is not a key of the file.

```yaml title="routers/dynamo-round-robin.yaml"
--8<-- "catalog/routers/dynamo-round-robin.yaml"
```

## Keys

<!-- fields: routertable.Entry -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `RouterMapping`. |
| `router` | string | yes | The router the run was measured behind, in lower case, for example `dynamo`. Never `llm-d`, which BLIS hosts. |
| `mode` | string | yes | The router's mode or algorithm as the run's configuration names it. When the configuration sets none, the mode the router's code defaults to, and a citation says so. |
| `session_affinity_ttl_s` | integer | no | The router's session-affinity idle timeout in seconds, when the configuration sets one. At least 1. Two runs that differ in it are different keys. |
| `revisions` | list of strings | yes | The builds of the router the run used for which the entry is established, each a full 40-character commit, sorted, each once. A mapping is a claim about the behaviour of a build, so an entry shown for one build is not inherited by another with the same mode: a run on a build not listed has no mapping and stays `router_unsupported` (design §11.4). Every `router`-side citation is at one of them or at a commit of a build, and each has a citation for the `mode` and one for the `policy`. |
| `builds` | list | no | The builds a recipe names by version, not by commit (a wheel such as `1.3.0.dev20260601`), each with the commits it could have been made from and the files that decide the mode, below. |
| `policy` | string | yes | A policy llm-d's router offers, or `none`. |
| `reason` | string | yes | What the router's mode and the policy do alike and differently; for `none`, why no policy is close enough. |
| `difference` | list | for a policy other than `none`; else no | Each way the mode and the policy differ, one statement per aspect, each citing the code that shows it, below. |
| `bound` | mapping | for a policy other than `none`; else no | How far the choice of policy moves each figure on BLIS, measured, with its regime and evidence, below. |
| `evidence` | list | yes | Citations of both routers' code or documentation, below. |

The key of an entry is the router, the mode and the session-affinity setting. The entry's file is
named `<router>-<mode>`, with `-affinity-<seconds>` when the setting is set, and the validator
refuses another name, so one key cannot be stated by two files, in a catalog or in an overlay.

### Lookup by revision

`routertable.Resolve` (and `Data.LookupRouter`) takes the router, the mode, the session-affinity
setting and the router build the run used, and returns the entry with one of four outcomes.
`mapped`: the entry covers the build and names a policy; only this scores the point, at G2
(`router_differs`). `no_policy`: it covers the build and says no policy is close enough.
`revision_not_covered`: an entry has the key and does not list the build, so what it shows is
not shown for this one. `no_entry`: no entry has the key. The last three are G5
`router_unsupported`. A point that relies on an entry carries the build it used in the
`revision` of its `router_differs`, and `Data.CheckRouterSubstitutions` refuses one whose entry
is missing, is another router's, does not cover the build, or maps to another policy.

`routertable.ResolveBuild` (and `Data.LookupRouterBuild`) is the same lookup for a run whose recipe
names the build by version. A recipe pins Dynamo by a wheel such as `1.3.0.dev20260601`, which is
no commit, so the entry lists the build under `builds` with every commit it could have been made
from; the outcomes are the same four, and a version the entry does not list, or a recipe that
states none, is `revision_not_covered`. `Resolve` by commit also finds a commit that lies in a
listed build. A point that relies on a build names `Build.Revision()`, the newest commit of its
window, in the `revision` of its `router_differs`.

#### `builds`

A nightly wheel carries the UTC date it was built on and is made from the head of the main branch
when the run was triggered; the hour is not in the version. The window of a nightly is therefore
every first-parent commit of main from the head at the start of that date to the head at its end.
A release wheel is the one commit of its tag. The entry covers the build only if the files that
decide the mode have one git blob identifier at every commit of the window (`git rev-parse
<commit>:<path>`); every `router`-side citation at one of those commits is of one of those files,
so the passage read at one commit is the passage at all of them. Where a file changes inside a
window, the build is not listed and a run on it stays `router_unsupported`. To redo a window:
`git log --first-parent --format='%H %ct' main`, take the commits from the head before the date's
00:00 UTC to the head before the next date's, and print each file's blob with `git rev-parse`.

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | The version as the recipe states it, for example `1.3.0.dev20260601` or `1.2.1`. Each build of an entry once. |
| `source` | string | yes | `nightly`: a development wheel, made from main on the date in its version. `release`: a release wheel, made from the commit of its tag. |
| `window` | string | yes | In words, how the commits were found, so a reader can redo it from the router's history. |
| `commits` | list of strings | yes | Every commit the build could have been made from, each a full 40-character commit, sorted, each once. A release has one. A commit is in one build of an entry only. |
| `files` | list | yes | The files that decide the mode, each a `path` from the router repository's root and its git `blob` identifier (40 lower-case hex digits), the same at every commit of `commits`. |

The committed `dynamo-round-robin` entry covers Dynamo `1.2.0.dev20260426`, `1.2.1`,
`1.3.0.dev20260601` and `1.3.0.dev20260614`, and the commit `1682167d3c97f0fc8cd405dee51f17288eeaeb55`
(26 April 2026). In the window of `1.3.0.dev20260614` the file `push_router.rs` has two blobs. They
differ only in what the rule set `rust-logging-only` removes: logging statements, a binding that
only carries a value to a removed statement, an unread element of a tuple binding, and blank
lines. The entry lists the second blob as a `variant` of the first, each with the SHA-256 of its
normal form, and the validator requires the two digests to be equal. The digests are produced by
`catalog/scripts/router_normal_form.py` on a clone of the router's repository, which anyone can
rerun; a difference left after the rules is a difference in what the router computes, and the
build is then not covered. A run on any other Dynamo build stays `router_unsupported` until an
entry cites it.

The `vllm-router-consistent_hash` entry covers vLLM Router `0.1.14`, a release (the commit of the
project's tag `v0.1.14`), the build InferenceX installs for its AgentX runs. It maps to `none`, so
a run behind it resolves to `no_policy`: the table knows the router and says why no policy of
llm-d's router is close enough. A run on another build of vLLM Router, such as a dated nightly,
is `revision_not_covered`.

Policies:

--8<-- "docs/contracts/generated/enum/routertable.Policy.md"

### `evidence`

<!-- fields: routertable.Citation -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `side` | string | yes | Whose code it is, below. |
| `field` | string | yes | The field of the entry it supports, below. |
| `repository` | string | yes | `<owner>/<name>` on GitHub. On the `llm-d` side, `llm-d/llm-d-router` or `inference-sim/llm-d-router`; on the `router` side, not those. |
| `commit` | string | yes | The full 40-character commit the passage is read at. A branch or a tag is not a pinned revision. |
| `path` | string | yes | The file, from the repository's root. For a policy other than `none`, one `llm-d` citation shows the behaviour the policy is matched to, at the commit it names; the contract keeps no list of files. |
| `lines` | string | yes | A line number, or a range `<first>-<last>`, at `commit`. |
| `claim` | string | yes | What the passage shows. |

Sides:

--8<-- "docs/contracts/generated/enum/routertable.Side.md"

`router`
:   The router the run used.
`llm-d`
:   llm-d's router. It supports the policy only.

Fields:

--8<-- "docs/contracts/generated/enum/routertable.Field.md"

Each field the entry states has a citation (SUB-18): the router's own code for its `mode` and,
when set, its `session_affinity_ttl_s`; and, for the `policy`, a passage of the router that shows
the behaviour it is matched to and a passage of llm-d's router that shows the policy does it.

### `difference`

<!-- fields: routertable.Difference -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `aspect` | string | yes | What differs, as a lower-case slug, for example `counter`, `tie-break` or `scale`. Each aspect once. |
| `text` | string | yes | How the router's mode and the policy differ in that aspect, and what BLIS does instead. It states a fact of the code, or says that an effect is not measured. |
| `evidence` | list of integers | yes | The positions (from 0) in the entry's `evidence` list of the citations the statement rests on. At least one, each once. |

A [point record](point-record.md) carries the same statements without `evidence`, because a
position means nothing outside the entry:

<!-- fields: routertable.Statement -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `aspect` | string | yes | What differs, as above. |
| `text` | string | yes | How the router's mode and the policy differ in that aspect. |

A mapping is a claim of nearness, so the entry says how near: every way the two differ that the
code shows, and for each the passages that show it. A point scored with the entry copies the
statements (without the positions, which mean nothing outside the entry) into its
[`router_differs` substitution](point-record.md).

### `bound`

A bound is a sensitivity, not an error. It says how far BLIS's own prediction of a figure moves
when the policy changes, so it says what a wrong guess of the router's mode could cost. It does
not say how far BLIS is from the router's real behaviour, which no run measured.

A bound is measured on BLIS, so it is a fact about the two policies and the workload, not about a
build of the router. The entry states it once for every build in its `revisions`, as it states
the differences, which are facts of the code at those builds. A difference or a bound that holds
at one build only would need an entry of its own, which the key (router, mode, affinity) does not
allow yet.

<!-- fields: routertable.Bound -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `evidence` | mapping | yes | The measurement the ratios come from, below. |
| `ratios` | list | yes | The limits: at least one, each for one figure, one set of alternatives and one regime. |
| `not_measured` | list of strings | yes | What the bound does not cover, one sentence each. At least one: a bound that claims to cover everything is not a measured bound. |

<!-- fields: routertable.BoundEvidence -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `file` | string | yes | The CSV of runs, from the repository's root. The ratios are recomputed from it. |
| `sha256` | string | yes | The SHA-256 of the file, 64 lower-case hex digits, so a ratio cites the bytes it was read from. |
| `producer` | string | yes | `<owner>/<name>` of the repository BLIS was built from for the runs. |
| `commit` | string | yes | The full 40-character commit of the producer. |
| `command` | string | yes | The command that reproduces the file. |

<!-- fields: routertable.Ratio -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `metric` | string | yes | The [metric registry](metrics.md) identifier of the figure, for example `ttft_p50_ms`. |
| `against` | list of strings | yes | The alternatives the policy was compared with, each by the name of its plugin composition in the evidence file's `policy` column. Sorted, each once, none the entry's own policy. |
| `regime` | mapping | yes | Where the ratio was measured, below. |
| `max_ratio` | number | yes | The largest, over every cell of the regime and every alternative, of the larger of the two means over the smaller, each mean taken over the seeds. At least 1, rounded up to two decimals so it never understates. |

<!-- fields: routertable.Regime -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `workload` | string | yes | The kind of workload, below. |
| `concurrencies` | list of integers | yes | The closed-loop client counts the cells ran, ascending, each once, each at least 1. A point matches only a count in the list: nothing is interpolated between two, because time to first token has a sharp knee between them. |
| `model` | string | yes | The catalog model the runs used. |
| `hardware` | string | yes | The catalog hardware the runs used. |
| `engine_version` | string | yes | The engine release the runs simulated, as the scenario's `engine_version` states it. |
| `fabric` | string | yes | The catalog network between the pools, the scenario's `cluster.fabric`. |
| `prefill` | mapping | yes | The parallelism of one prefill instance: `tp`, `pp`, `dp` (each at least 1) and `expert_parallel`. |
| `decode` | mapping | yes | The same for one decode instance. |
| `shapes` | list of strings | yes | The pool shapes the cells cover, each `<n>P<m>D`: n prefill and m decode instances. Sorted, each once. |

### Regime of a point

<!-- fields: routertable.Point -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `workload` | string | yes | The workload kind of the point. |
| `concurrency` | integer | yes | Its closed-loop client count. At least 1. |
| `model` | string | yes | The catalog model it simulates. |
| `hardware` | string | yes | The catalog hardware. |
| `engine_version` | string | yes | The engine release the scoring simulator runs. |
| `fabric` | string | yes | The catalog network between the pools. |
| `shape` | string | yes | Its pool shape, `<n>P<m>D`. |
| `prefill` | mapping | yes | The parallelism of one prefill instance, as for a regime. |
| `decode` | mapping | yes | The same for one decode instance. |

A router_differs substitution carries this regime, and the part of the entry's bound that covers
it (`Bound.For`): `load.Data.CheckRouterSubstitutions` recomputes that part from the entry and
refuses a bound that omits a covering ratio, carries one for another regime, or is carried for a
point the entry has no bound for.

<!-- fields: routertable.Parallel -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tp` | integer | yes | Tensor-parallel width of one instance. At least 1. |
| `pp` | integer | yes | Pipeline-parallel width. At least 1. |
| `dp` | integer | yes | Data-parallel width. At least 1. |
| `expert_parallel` | boolean | yes | Whether the instance splits its experts across its ranks. |

Workload kinds:

--8<-- "docs/contracts/generated/enum/routertable.WorkloadKind.md"

`balanced`
:   Closed loop, every request 1,024 prompt tokens and 1,024 output tokens, no prefix shared between requests.
`prefill-bound`
:   Closed loop, 8,192 prompt tokens and 16 output tokens.
`decode-bound`
:   Closed loop, 256 prompt tokens and 2,048 output tokens.

A regime is exact. `Entry.BoundFor` takes the point's workload, client count, model, hardware, engine
release, fabric, instance layouts and pool shape and returns the ratios whose regime contains all
of them; with none, the outcome is `regime_not_covered` and the point stays `router_unsupported`
with the reason "no bound for this regime". A point record that carries a router bound whose regime
does not list the point's client count is refused. The ratios are the sensitivity of BLIS's own
figures to the choice of policy, not the error of standing a policy in for the router.

The list is closed because a bound is a statement about the workloads it names and no others:
a kind joins it with the measurement that covers it. How the ratios are measured, how to
reproduce them and what they leave out is in the [router sensitivity guide](../guides/router-sensitivity.md).

## What the validator refuses

An entry with no evidence; one that lacks a citation for a field it states, or for either router's
side of the policy; a policy llm-d's router does not offer; a `none` or a match with no reason;
the router `llm-d`; an `llm-d` citation of another repository, a `router` citation of llm-d's, or an `llm-d` citation that names no file defining the policy; a file name that is not the entry's key; no revisions, a revision that is not a full commit, repeated or out of order, a `router` citation at a commit that is not one of the revisions, and a revision with no `router` citation of its own for the mode or the behaviour; a citation whose commit is not a full commit, whose path climbs out of the
repository or whose lines run backward; a citation repeated.

For a policy other than `none`: no `difference`; an aspect that is empty, not a slug or stated twice; a statement with no text or
no citation, or one that names a position outside the evidence list or names one twice; no
`bound`; a bound with no ratios, no `not_measured`, a file path that climbs, a hash that is not 64
lower-case hex digits, a producer that is not `<owner>/<name>`, a commit that is not a full commit or
no command; a ratio whose metric is not in the metric registry, that names no alternative or its
own policy, whose alternatives are unsorted or repeated, whose value is not a finite number of at
least 1 with two decimals, whose workload is not a kind above, whose client counts are empty, below 1 or not ascending, whose model, hardware, engine
release, fabric or either instance layout is missing, whose shapes are missing, malformed or unsorted; a
ratio stated twice for one regime. For `none`: a `difference` or a `bound`, because there is no
stand-in policy to differ from or to bound. The catalog's tests also recompute every ratio from
the evidence file and refuse one that is not the file's.
