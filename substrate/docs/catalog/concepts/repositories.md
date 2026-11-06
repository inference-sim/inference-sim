# The catalog in BLIS

BLIS predicts how an LLM serving deployment will perform. The catalog is one directory of blis-substrate, the repository that holds everything that is true whatever engine runs. Its parts are released together, at one tag:

| Directory | Owns | Holds |
| :--- | :--- | :--- |
| `contracts/` | the formats | Go types and loaders for every document the others hold, and the interfaces between the Go repositories. |
| `validate/` | the checks | The validators for the catalog and the registry, and their commands. |
| **`catalog/`** | declared facts | Models, chips, fabrics, workloads, storage tiers. YAML and JSON. |
| `registry/` | learned numbers | Coefficients a cost model fits or assumes, each with how it was obtained and the scope it applies to. |
| `kernel/` | pricing | The cost model: from a graph, a chip, coefficients and a deployment to a step time. |

[inference-sim](https://github.com/inference-sim/inference-sim), the discrete-event simulator, imports the cost model and the embedded catalog and registry, schedules requests, and calls the cost model once per step.

<figure markdown="1">
<div class="cf-scroll"><svg class="cf-figure" viewBox="0 0 720 430" role="img" aria-label="The directories of blis-substrate and how inference-sim uses them" xmlns="http://www.w3.org/2000/svg">
<title>The directories of blis-substrate and how inference-sim uses them</title>
<defs><marker id="cf-arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="cf-arrow" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<rect class="cf-box" x="10" y="8" width="700" height="300" rx="6"/>
<text class="cf-sub" x="24" y="28">blis-substrate: one repository, one release</text>
<path class="cf-edge" d="M480,84 L560,128" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="530" y="100">validates</text>
<path class="cf-edge" d="M440,84 L190,128" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="300" y="100" text-anchor="end">validates</text>
<path class="cf-edge" d="M200,84 L330,218" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="232" y="150" text-anchor="end">types and loaders</text>
<path class="cf-edge" d="M170,172 L300,218" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="222" y="212" text-anchor="end">graphs, chips,</text>
<text class="cf-edge-label" x="222" y="225" text-anchor="end">fabrics, storage</text>
<path class="cf-edge" d="M550,172 L420,218" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="498" y="214">coefficients</text>
<path class="cf-edge" d="M360,262 L360,338" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="368" y="300">step time; embedded catalog and registry</text>
<rect class="cf-box" x="100" y="40" width="200" height="44" rx="4"/>
<text class="cf-name" x="200" y="59" text-anchor="middle">contracts/</text>
<text class="cf-sub" x="200" y="74" text-anchor="middle">the formats</text>
<rect class="cf-box" x="380" y="40" width="200" height="44" rx="4"/>
<text class="cf-name" x="480" y="59" text-anchor="middle">validate/</text>
<text class="cf-sub" x="480" y="74" text-anchor="middle">the checks</text>
<rect class="cf-box-own" x="40" y="128" width="200" height="44" rx="4"/>
<text class="cf-name" x="140" y="147" text-anchor="middle">catalog/</text>
<text class="cf-sub" x="140" y="162" text-anchor="middle">declared facts</text>
<rect class="cf-box" x="480" y="128" width="200" height="44" rx="4"/>
<text class="cf-name" x="580" y="147" text-anchor="middle">registry/</text>
<text class="cf-sub" x="580" y="162" text-anchor="middle">learned numbers</text>
<rect class="cf-box" x="260" y="218" width="200" height="44" rx="4"/>
<text class="cf-name" x="360" y="237" text-anchor="middle">kernel/</text>
<text class="cf-sub" x="360" y="252" text-anchor="middle">pricing</text>
<rect class="cf-box" x="260" y="338" width="200" height="44" rx="4"/>
<text class="cf-name" x="360" y="357" text-anchor="middle">inference-sim</text>
<text class="cf-sub" x="360" y="372" text-anchor="middle">simulation</text>
<text class="cf-sub" x="360" y="412" text-anchor="middle">a scenario, supplied with each run, states the deployment</text>
</svg></div>
<figcaption>Each arrow runs from what is provided to what uses it. The catalog and the registry hold data, checked by the validators in the same repository. The cost model reads both through the contracts' loaders. inference-sim imports the cost model and the data embedded in the release, and calls the cost model once per simulated step. Deployment choices are in the scenario the user supplies with a run.</figcaption>
</figure>

## Why split it this way

The parts are separate directories because they change for different reasons and are reviewed by different people. They are one repository because a change often needs several of them at once, and in one repository that is one pull request, one review and one release.

- **Formats and contents.** The contracts say what a valid chip file is, and the catalog says what an H100 is. A new chip changes only the catalog. A new field changes the contracts, and the catalog adopts it in the same pull request.
- **Declared and learned.** A datasheet figure is settled once it is written down correctly. A coefficient can change when new measurements arrive, and it applies only within its scope. Keeping them apart lets each coefficient carry its scope and method, and keeps the catalog free of numbers that need them. [Declared, learned, chosen](declared-facts.md) develops this.
- **Data and pricing.** The cost model reads models and chips from the catalog, and its library code contains no model names: it dispatches on the operations in a graph. A new model therefore reaches the cost model as data, without a code change.
- **Pricing and simulation.** The simulator decides which requests form a batch. The cost model decides what that batch costs. Either can be replaced without touching the other.

## What the catalog depends on

Nothing at run time: the catalog contains no code that a simulation executes. Its one script, `catalog/scripts/derive_graph.py`, translates a vendor config into a model graph and runs when an entry is added. The site's reference pages are built by `docs/hooks/catalog_pages.py`. Neither runs during a simulation.

The catalog is validated by the document types in `contracts/` and the validators in `validate/`, at the same commit. CI runs `go run ./validate/cmd/validate-catalog catalog` on every change, so the catalog and the code that reads it cannot drift apart.

## Where a new model lands

Either way, the change is one pull request in blis-substrate.

| What the model needs | What the pull request changes |
| :--- | :--- |
| Only operations the graph format already defines | the catalog alone: a new `models/<name>/`, and a deriver handler if the architecture is new |
| A property no existing field can express | a new field in `contracts/spec` with its validator, the cost model's reading of it in `kernel/`, new coefficients in `registry/` if it needs them, and the catalog entry, together |

In neither case does the simulator's code change; in the second, inference-sim adopts the new substrate release.
