# Using the registry

The registry is read from disk, by path, at run time. Nothing is installed and nothing is
fetched over the network: a consumer is given the root of a clone and opens
`coefficients/<set>.yaml` beneath it.

## Get a copy

Clone a release, not `main`. A release fixes every value, and an estimate is reproducible
only if the coefficients behind it are.

```sh
git clone --branch <!-- registry:release --> --depth 1 https://github.com/inference-sim/blis-registry.git
```

<!-- registry:release --> is the newest release at the time this page was built.
[Releases and pinning](releases.md) explains what a version number does and does not
promise.

## Run BLIS with it

The simulator prices steps through blis-latency-kernel when run with
`--latency-model blis-latency-kernel`. That path is on the `kernel-exclusive` branch of
the `sriumcp/inference-sim` fork, open as
[inference-sim#1851](https://github.com/inference-sim/inference-sim/pull/1851).
It reads the deployment from a scenario file, the model and hardware from a catalog
clone, and the coefficients from a registry clone:

```sh
./blis run --latency-model blis-latency-kernel \
    --scenarios path/to/scenarios --scenario kimi-k3-h100-nospec.yaml \
    --catalog path/to/blis-catalog \
    --registry path/to/blis-registry
```

The scenario's `coefficients:` list names the sets to load, and its `cluster.hardware`
selects the entries that apply. `--registry` is required on this path; the run stops
with an error rather than fall back to other coefficients.

## Use it from Go

`blis-latency-kernel` resolves a scenario file against the three roots and returns a
kernel ready to price steps:

```go
import latencykernel "github.com/inference-sim/blis-latency-kernel"

k, err := latencykernel.Open("kimi-k3-h100-nospec.yaml", latencykernel.Repos{
    Scenarios: "path/to/scenarios",
    Catalog:   "path/to/blis-catalog",
    Registry:  "path/to/blis-registry",
})
```

To read one set without the kernel, load and validate it with blis-schemas, as the
kernel does:

```go
import blisschemas "github.com/inference-sim/blis-schemas"

set, err := blisschemas.LoadCoefficientSet("path/to/blis-registry/coefficients/cost-model-attention.yaml")
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

with open("coefficients/cost-model-attention.yaml") as f:
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
