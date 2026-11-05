# Using the registry

The registry is data: one YAML file per coefficient set under `registry/coefficients/`.
Each blis-substrate release embeds it, so a run of a released simulator needs no copy of
its own.

## Run BLIS with it

A run that names no registry uses the copy embedded in the release. To use another, such as
a candidate with refitted values, name its directory; the run uses it instead of the
embedded copy and records which one it used, with its content hash (design §3.3):

```sh
blis run --scenario path/to/scenario.yaml --registry path/to/registry
```

The scenario's `coefficients:` list names the sets to load, and its `cluster.hardware`
selects the entries that apply.

## Use it from Go

`kernel.Open` resolves a scenario against the catalog and registry and returns a kernel
ready to price steps. It uses the same loader and checks as the simulator.

To read one set without the kernel, load and validate it with the `contracts` package, as
the kernel does:

```go
import "github.com/inference-sim/blis-substrate/contracts"

set, err := contracts.LoadCoefficientSet("registry/coefficients/cost-model-attention.yaml")
if err != nil {
    return err // unreadable, or an unknown field
}
if p := set.Validate(); !p.OK() {
    return p // every finding, each naming the entry it is about
}
for _, e := range set.Coefficients {
    fmt.Println(e.Name, e.Scope.Hardware, e.Value, e.Units, e.Method)
}
```

## Read it from anything else

The files are plain YAML. Each item in `coefficients` is a map with a single key, the
coefficient's name:

```python
import yaml

with open("registry/coefficients/cost-model-attention.yaml") as f:
    doc = yaml.safe_load(f)

for item in doc["coefficients"]:
    (name, entry), = item.items()
    print(name, entry["scope"]["hardware"], entry["value"], entry["units"], entry["method"])
```

A name can occur more than once in a set, once per scope. A reader that builds a
dictionary keyed by name alone will keep one part's value and silently drop the rest.
Key by name and scope, or filter by scope first, as the kernel does.

## Check where a number came from

Every value in the [coefficient set reference](../reference/index.md) links to its entry
in the committed file at the revision the page was built from. The entry carries the
citation, the rationale and the scope. Follow the citation to the dataset; the
[reproduction guide](../reproducing-coefficients.md) gives the command that regenerates
it.
