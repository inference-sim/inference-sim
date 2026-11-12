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
| `revisions` | list of strings | yes | The builds of the router the run used for which the entry is established, each a full 40-character commit, sorted, each once. A mapping is a claim about the behaviour of a build, so an entry shown for one build is not inherited by another with the same mode: a run on a build not listed has no mapping and stays `router_unsupported` (design §11.4). Every `router`-side citation is at one of them, and each has a citation for the `mode` and one for the `policy`. |
| `policy` | string | yes | A policy llm-d's router offers, or `none`. |
| `reason` | string | yes | What the router's mode and the policy do alike and differently; for `none`, why no policy is close enough. |
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

The committed `dynamo-round-robin` entry covers one build of Dynamo, commit
`1682167d3c97f0fc8cd405dee51f17288eeaeb55` (26 April 2026), the build the April 2026 recipes
pin. A run on a later Dynamo build stays `router_unsupported` until an entry cites it.

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

## What the validator refuses

An entry with no evidence; one that lacks a citation for a field it states, or for either router's
side of the policy; a policy llm-d's router does not offer; a `none` or a match with no reason;
the router `llm-d`; an `llm-d` citation of another repository, a `router` citation of llm-d's, or an `llm-d` citation that names no file defining the policy; a file name that is not the entry's key; no revisions, a revision that is not a full commit, repeated or out of order, a `router` citation at a commit that is not one of the revisions, and a revision with no `router` citation of its own for the mode or the behaviour; a citation whose commit is not a full commit, whose path climbs out of the
repository or whose lines run backward; a citation repeated.
