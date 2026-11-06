package load

import (
	"fmt"
	"slices"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/validate/dataset"
)

// applyCatalogOverlay applies a catalog overlay's files to files, by the catalog's
// identities (design §3.3): a model is its whole directory, so an overlay's models/<name>/
// replaces every data file of that model; any other file replaces the file at its path, or
// is added. An overlay path that differs from a base path only in letter case is refused:
// the two would sit side by side, the base entry would stay in force, and the overlay's hash
// would still say it applied.
func applyCatalogOverlay(files, overlay map[string][]byte) error {
	folded := map[string]string{}
	for base := range files {
		folded[strings.ToLower(base)] = base
	}
	var clash []string
	for rel := range overlay {
		if base, ok := folded[strings.ToLower(rel)]; ok && base != rel {
			clash = append(clash, fmt.Sprintf("%s and %s differ only in letter case; an overlay replaces a file by its exact path", rel, base))
		}
		if dir, ok := modelDir(rel); ok {
			for base := range files {
				if bdir, ok := modelDir(base); ok && bdir != dir && strings.EqualFold(bdir, dir) {
					clash = append(clash, fmt.Sprintf("%s and %s differ only in letter case; an overlay replaces a model by its exact name", dir, bdir))
					break
				}
			}
		}
	}
	if len(clash) > 0 {
		slices.Sort(clash)
		return fmt.Errorf("a catalog overlay: %s", strings.Join(slices.Compact(clash), "; "))
	}
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
	return nil
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

// checkSetLevelFields refuses an overlay that states a different role or caveats for a set
// the registry already has. The merge applies profiles and coefficients only, so it would
// drop those fields while the hash and provenance said the overlay applied. An overlay that
// states no caveats leaves the set's own.
func checkSetLevelFields(r, ov *dataset.Registry) error {
	for stem, os := range ov.Acceptance {
		if base, ok := r.Acceptance[stem]; ok && len(os.Caveats) > 0 && !slices.Equal(os.Caveats, base.Caveats) {
			return fmt.Errorf("the registry overlay's acceptance/%s.yaml changes the set's caveats; an overlay adds or replaces profiles, it cannot change set-level fields", stem)
		}
	}
	for stem, os := range ov.Routing {
		base, ok := r.Routing[stem]
		if !ok {
			continue
		}
		if os.Role != base.Role {
			return fmt.Errorf("the registry overlay's routing/%s.yaml changes the set's role from %q to %q; an overlay adds or replaces profiles, it cannot change set-level fields", stem, base.Role, os.Role)
		}
		if len(os.Caveats) > 0 && !slices.Equal(os.Caveats, base.Caveats) {
			return fmt.Errorf("the registry overlay's routing/%s.yaml changes the set's caveats; an overlay adds or replaces profiles, it cannot change set-level fields", stem)
		}
	}
	return nil
}
