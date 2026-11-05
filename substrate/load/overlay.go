package load

import (
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/validate/dataset"
)

// applyCatalogOverlay applies a catalog overlay's files to files, by the catalog's
// identities (design §3.3): a model is its whole directory, so an overlay's models/<name>/
// replaces every data file of that model; any other file replaces the file at its path, or
// is added.
func applyCatalogOverlay(files, overlay map[string][]byte) {
	for rel := range overlay {
		if dir, ok := modelDir(rel); ok {
			for base := range files {
				if strings.HasPrefix(base, dir) {
					delete(files, base)
				}
			}
		}
	}
	for rel, b := range overlay {
		files[rel] = b
	}
}

// modelDir returns "models/<name>/" for a file in a model directory.
func modelDir(rel string) (string, bool) {
	rest, ok := strings.CutPrefix(rel, "models/")
	if !ok {
		return "", false
	}
	name, _, ok := strings.Cut(rest, "/")
	return "models/" + name + "/", ok
}

// coefficientID is one coefficient's identity: its name, its scope and its method, within
// its set (design §3.3).
func coefficientID(e coefficient.Entry) string {
	return e.Name + "\x00" + e.Scope.Key() + "\x00" + string(e.Method)
}

// applyRegistryOverlay merges an overlay's sets into r. A coefficient with the identity of
// one in the same set replaces it in place; any other is appended to its set, and a set the
// registry does not have is added whole. Acceptance and routing profiles are identified the
// same way, by their set and their key.
func applyRegistryOverlay(r, ov *dataset.Registry) {
	for stem, os := range ov.Sets {
		base, ok := r.Sets[stem]
		if !ok {
			r.Sets[stem] = os
			continue
		}
		merged := &coefficient.Set{Kind: base.Kind, Name: base.Name,
			Coefficients: append([]coefficient.Entry(nil), base.Coefficients...)}
		at := map[string]int{}
		for i, e := range merged.Coefficients {
			at[coefficientID(e)] = i
		}
		for _, e := range os.Coefficients {
			if i, ok := at[coefficientID(e)]; ok {
				merged.Coefficients[i] = e
				continue
			}
			at[coefficientID(e)] = len(merged.Coefficients)
			merged.Coefficients = append(merged.Coefficients, e)
		}
		r.Sets[stem] = merged
	}
	for stem, os := range ov.Acceptance {
		base, ok := r.Acceptance[stem]
		if !ok {
			r.Acceptance[stem] = os
			continue
		}
		merged := *base
		merged.Profiles = append([]acceptance.Profile(nil), base.Profiles...)
		at := map[acceptance.Key]int{}
		for i, p := range merged.Profiles {
			at[p.Key()] = i
		}
		for _, p := range os.Profiles {
			if i, ok := at[p.Key()]; ok {
				merged.Profiles[i] = p
				continue
			}
			at[p.Key()] = len(merged.Profiles)
			merged.Profiles = append(merged.Profiles, p)
		}
		r.Acceptance[stem] = &merged
	}
	for stem, os := range ov.Routing {
		base, ok := r.Routing[stem]
		if !ok {
			r.Routing[stem] = os
			continue
		}
		merged := *base
		merged.Profiles = append([]routing.Profile(nil), base.Profiles...)
		at := map[routing.Key]int{}
		for i, p := range merged.Profiles {
			at[p.Key()] = i
		}
		for _, p := range os.Profiles {
			if i, ok := at[p.Key()]; ok {
				merged.Profiles[i] = p
				continue
			}
			at[p.Key()] = len(merged.Profiles)
			merged.Profiles = append(merged.Profiles, p)
		}
		r.Routing[stem] = &merged
	}
}
