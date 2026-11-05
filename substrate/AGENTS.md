# Agent instructions for blis-substrate

These rules apply to every change to this repository, by a person or an agent. They are written
here so that an agent reads them first. The rules come from the BLIS target design; each names
its invariant identifier.

## What belongs here

blis-substrate holds what is true whatever engine runs (design §2.2). Before adding a fact, ask
whether a second engine would need the same edit. If it would not, the change belongs in an
engine repository.

- Outside tests, no engine name, engine version or engine policy (X2).
- The cost model resolves nothing: every default, backend choice, padding and capture decision
  arrives in the plan or the step shape (SUB-6).
- The catalog holds declared facts, each with a citation. A fitted or assumed value is a
  coefficient, and belongs in the registry (SUB-3, SUB-18).
- Every coefficient is written by its family's writer script, never by hand (SUB-4).
- Packages under `contracts/` import only the Go standard library, the YAML parser and each
  other, and change only compatibly (SUB-1).
- One Go module: no `go.mod` below the root (SUB-13).

## Before a pull request

Run the checks for every directory the change touches:

```sh
gofmt -l . && go vet ./... && go test ./...
go run ./validate/cmd/validate-catalog catalog
go run ./validate/cmd/validate-registry registry
(cd catalog && python3 scripts/derive_graph.py --check && python3 -m pytest tests -q)
(cd registry && python3 -m pytest validator -q)
mkdocs build --strict
```

- Every behaviour change has a test that fails without it (X13).
- A new or changed data field ships with its validator, a failing example for each way it can
  be wrong, and regenerated reference documentation: `go run ./tools/docgen` (SUB-2).
- Never commit third-party data; a derived value cites its source by URL and commit or hash
  (X12). Never write a private name in a file, commit or pull request (X12).
- A commit written with an agent's help says so in one `Assisted-by:` line. The person who takes
  responsibility for it adds the DCO `Signed-off-by:` line; an agent never signs off (X20).

## Agents may not

Edit a generated file by hand; weaken, skip or delete a check; change a success criterion after
reading held-out data; answer a review question; approve a merge; cut a release (design §13.7).
