# The catalog in BLIS

BLIS predicts how an LLM serving deployment will perform. It consists of five repositories, each owning one kind of thing:

| Repository | Owns | Holds |
| :--- | :--- | :--- |
| [blis-schemas](https://github.com/inference-sim/blis-schemas) | the formats | Go types and validators for every document the others hold, and per-engine rule packs. |
| **blis-catalog** | declared facts | Models, chips, fabrics, workloads, storage tiers. YAML and JSON. |
| [blis-registry](https://github.com/inference-sim/blis-registry) | learned numbers | Coefficients a cost model fits or assumes, each with how it was obtained and the scope it applies to. |
| [blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel) | pricing | The cost model: from a graph, a chip, coefficients and a deployment to a step time. |
| [inference-sim](https://github.com/inference-sim/inference-sim) | simulation | The discrete-event simulator that schedules requests and calls the kernel once per step. |

<figure markdown="1">
<div class="cf-scroll"><svg class="cf-figure" viewBox="0 0 720 410" role="img" aria-label="How the five BLIS repositories depend on one another" xmlns="http://www.w3.org/2000/svg">
<title>How the five BLIS repositories depend on one another</title>
<defs><marker id="cf-arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="cf-arrow" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<path class="cf-edge" d="M300,62 L170,118" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="196" y="82" text-anchor="end">validates</text>
<path class="cf-edge" d="M420,62 L550,118" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="524" y="82">validates</text>
<path class="cf-edge" d="M360,62 L360,218" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="368" y="146">types and loaders</text>
<path class="cf-edge" d="M170,162 L300,218" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="222" y="208" text-anchor="end">graphs, chips,</text>
<text class="cf-edge-label" x="222" y="221" text-anchor="end">fabrics, storage</text>
<path class="cf-edge" d="M550,162 L420,218" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="498" y="214">coefficients</text>
<path class="cf-edge" d="M360,262 L360,318" marker-end="url(#cf-arrowhead)"/>
<text class="cf-edge-label" x="368" y="294">step time</text>
<rect class="cf-box" x="260" y="18" width="200" height="44" rx="4"/>
<text class="cf-name" x="360" y="37" text-anchor="middle">blis-schemas</text>
<text class="cf-sub" x="360" y="52" text-anchor="middle">the formats</text>
<rect class="cf-box-own" x="40" y="118" width="200" height="44" rx="4"/>
<text class="cf-name" x="140" y="137" text-anchor="middle">blis-catalog</text>
<text class="cf-sub" x="140" y="152" text-anchor="middle">declared facts</text>
<rect class="cf-box" x="480" y="118" width="200" height="44" rx="4"/>
<text class="cf-name" x="580" y="137" text-anchor="middle">blis-registry</text>
<text class="cf-sub" x="580" y="152" text-anchor="middle">learned numbers</text>
<rect class="cf-box" x="260" y="218" width="200" height="44" rx="4"/>
<text class="cf-name" x="360" y="237" text-anchor="middle">blis-latency-kernel</text>
<text class="cf-sub" x="360" y="252" text-anchor="middle">pricing</text>
<rect class="cf-box" x="260" y="318" width="200" height="44" rx="4"/>
<text class="cf-name" x="360" y="337" text-anchor="middle">inference-sim</text>
<text class="cf-sub" x="360" y="352" text-anchor="middle">simulation</text>
<text class="cf-sub" x="360" y="392" text-anchor="middle">a scenario, supplied with each run, states the deployment</text>
</svg></div>
<figcaption>Each arrow runs from what is provided to what uses it. The catalog and the registry hold data and are validated against the schemas. The kernel reads both through the schemas' loaders, and the simulator calls the kernel once per simulated step. Deployment choices are in the scenario the user supplies with a run.</figcaption>
</figure>

## Why split it this way

Each boundary separates things that change for different reasons and are owned by different people.

- **Formats and contents.** The schemas say what a valid chip file is, and the catalog says what an H100 is. A new chip changes only the catalog. A new field changes the schemas first, and the catalog adopts it by raising the validator version it pins.
- **Declared and learned.** A datasheet figure is settled once it is written down correctly. A coefficient can change when new measurements arrive, and it applies only within its scope. Keeping them apart lets each coefficient carry its scope and method, and keeps the catalog free of numbers that need them. [Declared, learned, chosen](declared-facts.md) develops this.
- **Data and pricing.** The kernel reads models and chips from the catalog, and its library code contains no model names: it dispatches on the operations in a graph. A new model therefore reaches the kernel as data, without a code change.
- **Pricing and simulation.** The simulator decides which requests form a batch. The kernel decides what that batch costs. Either can be replaced without touching the other.

## What the catalog depends on

Nothing at run time: the catalog contains no code that a simulation executes. Two scripts live here. `scripts/derive_graph.py` translates a vendor config into a model graph and runs when an entry is added. `scripts/docs/catalog_pages.py` builds this site. Neither runs during a simulation.

The catalog is validated by the document types in `contracts/` of the same repository. CI runs `validate/cmd/validate-catalog` against every commit, so the catalog and the code that reads it cannot drift apart.

## Where a new model lands

| What the model needs | Where the change lands |
| :--- | :--- |
| Only operations the schemas already define | the catalog alone: a new `models/<name>/`, and a deriver handler if the architecture is new |
| A property no existing field can express | blis-schemas first, then blis-latency-kernel, then the catalog |

In neither case does the simulator's code change; in the second, it adopts the new kernel and schema versions.
