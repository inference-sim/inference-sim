module github.com/inference-sim/blis-sglang

go 1.27.2

require (
	// blis-substrate at its m3/m31-integration head b6748ca9d757f149ba760cf821f96feed3bbf5bf (a commit pin, blis-design #22): inference-sim builds this module
	// against that substrate, so these tests run against the substrate that ships. v0.15.0 carries contracts/harness.
	github.com/inference-sim/blis-substrate v0.15.1-0.20261118162118-b6748ca9d757
	gopkg.in/yaml.v3 v3.0.1
)
