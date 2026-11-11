# Router table

<p class="lede">An entry of the router table maps one router BLIS does not host, in one mode,
to the policy of llm-d's router that stands closest to it, with the evidence for the match
(the design change in blis-design#1, a draft that amends §11.3 and §11.4 and has not merged).</p>

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
| `policy` | string | yes | A policy llm-d's router offers, or `none`. |
| `reason` | string | yes | What the router's mode and the policy do alike and differently; for `none`, why no policy is close enough. |
| `evidence` | list | yes | Citations of both routers' code or documentation, below. |

The key of an entry is the router, the mode and the session-affinity setting. The entry's file is
named `<router>-<mode>`, with `-affinity-<seconds>` when the setting is set, and the validator
refuses another name, so one key cannot be stated by two files, in a catalog or in an overlay.

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
| `path` | string | yes | The file, from the repository's root. For a policy other than `none`, one `llm-d` citation names a file that defines the policy (`routertable.Policies`). |
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
the router `llm-d`; an `llm-d` citation of another repository, a `router` citation of llm-d's, or an `llm-d` citation that names no file defining the policy; a file name that is not the entry's key; a citation whose commit is not a full commit, whose path climbs out of the
repository or whose lines run backward; a citation repeated.
