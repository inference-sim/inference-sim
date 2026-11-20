// Original to blis-sglang.

package blissglang

import (
	_ "embed"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/codepaths"
)

// CodePathsJSON is code_paths.json, the code-path table this build publishes (design §9.2, §12.4,
// Table DC): for the SGLang release this build models (Upstream), the functions its host, scheduler
// and graph-launch costs depend on, with a hash of each and the CodeID that follows. Its type is
// blis-substrate's contracts/spec/codepaths; substrate/tools/codepaths/regen.py writes it from the
// release's source, and a Go test and CI check it is current. The functions are named in code_paths/spec.json.
//
//go:embed code_paths.json
var CodePathsJSON []byte

// codePaths is the embedded table, parsed once when the package is initialised.
var codePaths, codePathsErr = codepaths.Parse("code_paths.json", CodePathsJSON)

// CodePaths returns the table, loaded through the substrate's strict loader, which recomputes
// every path's CodeID.
func CodePaths() (*codepaths.Table, error) { return codePaths, codePathsErr }

// CodeIDOf returns the CodeID of one code path of this release (coefficient.CodePathHost,
// coefficient.CodePathGraphs). It is what the lowering puts in plan.HostPlan.CodeID and
// plan.GraphPolicy.CodeID, so the cost model's lookup matches an engine-path coefficient only for
// the code that runs. It panics when the embedded table is unreadable or lists no such path: the
// build is wrong, and TestTheCodePathTableLoadsAndNamesThisRelease fails first.
func CodeIDOf(path string) string {
	if codePathsErr != nil {
		panic("blis-sglang: code_paths.json does not load: " + codePathsErr.Error())
	}
	id, ok := codePaths.CodeIDOf(path)
	if !ok {
		panic("blis-sglang: code_paths.json lists no path " + path)
	}
	return id
}
