# The field list

`fields.json` is the field list SGL-3 asks for. It holds every configuration field and every
environment variable the pinned SGLang release reads, each with a status, its evidence and an
estimation rule. Its form is the `FieldList` document of blis-substrate's `contracts/spec/fields`,
the one blis-vllm publishes.

| File | Written by | Holds |
|---|---|---|
| `fields.json` | `fields/gen_fields.py` | every name, with the status `fields/status.json` gives it |
| `fields/status.json` | people | each name's status, evidence and estimation rule, the dynamic reads and their names |
| `fields/declarations.json` | `fields/gen_fields.py --declarations` | what SGLang's own declarations state: each ServerArgs field's namespace, site, type, default, flags, choices and resolution metadata, and each environment variable's `Env*` class, default and read sites |
| `fields/value_classes.json` | `fields/gen_classes.py` | the value classes of each modelled or proven-inert field (VLM-3) |

The names come from SGLang's source at the commit `provenance.go` pins, read with Python's `ast`
module. SGLang is not imported. The rules for what counts as a field and as a read are in the
docstring of `fields/gen_fields.py`. In short:

- a configuration field is a field ServerArgs collects from the `A[...]` declarations of
  `python/sglang/srt/arg_groups/fields/*.py`, or a flag `ServerArgs.add_cli_args` registers by hand
  (`--config`);
- a field declared `Arg(no_cli=True)`, or declared with no `Arg` at all, has no command-line flag.
  It is listed all the same, since `Engine(**kwargs)` and `ServerArgs(...)` take it.
  `fields/declarations.json` records its flags as empty, and records any deprecated flag that
  stores into it (`--enable-dp-attention` into `enable_dp_attention`);
- an environment variable is an `Envs` field of `python/sglang/srt/environ.py`, a deprecated
  name it keeps, or a variable the serving process reads by a literal name.

## Regenerating

With an SGLang checkout at the pinned commit in `sglang-source`:

```sh
python3 -I fields/gen_fields.py sglang-source fields/status.json > fields.json
python3 -I fields/gen_fields.py sglang-source --declarations > fields/declarations.json
python3 -I fields/gen_classes.py sglang-source fields.json fields/declarations.json > fields/value_classes.json
```

CI runs each with `--check` and fails when a committed file differs. The generator also fails
when SGLang declares a name that `fields/status.json` gives no status, when `fields/status.json`
names something SGLang does not declare, and when a read whose name is not a literal is not listed
under `dynamic_reads`.

## Statuses

Every name starts `not_modelled`, because the Go port does not model any of them yet. Each cites
the open issue whose work models it (X28). When `config/` or an engine package models a field,
its entry becomes `modelled` with the traces that show it, and `fields/gen_classes.py` records its
value classes.

An unknown field, or a `not_modelled` field set away from its default, is refused by its
upstream name (X1).

## Fields with the rule leave_unset

Every `not_modelled` field with the rule `leave_unset` is named below. A scorer leaves such a
field at SGLang's default when a run's records do not state it. A value a run does state is never
replaced.

None yet.

End of the leave_unset list.
