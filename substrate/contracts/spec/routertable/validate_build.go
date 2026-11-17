package routertable

import (
	"fmt"
	"regexp"
	"strings"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
)

// validateBuilds checks the builds an entry names by version and returns the set of commits they
// cover. A build is a claim that the files which decide the mode are the same at every commit it
// could have been made from, so the entry must say which files and what they were, and its
// citations at those commits must be of those files and no others: a passage of a file the build
// does not vouch for says nothing of the commits in between.
func (e *Entry) validateBuilds(p *validate.Problems) map[string]bool {
	covered := map[string]bool{}
	names := map[string]bool{}
	for i, b := range e.Builds {
		at := fmt.Sprintf("builds[%d]", i)
		switch {
		case b.Name == "":
			p.Field(at+".name", "required: the version as the recipe states it, for example 1.3.0.dev20260601")
		case !token.MatchString(b.Name):
			p.Field(at+".name", "must be lower case letters, digits, '.', '_' or '-', got %q", b.Name)
		case names[b.Name]:
			p.Field(at+".name", "repeats the build %q", b.Name)
		}
		names[b.Name] = true
		switch b.Source {
		case BuildNightly, BuildRelease:
		default:
			p.Field(at+".source", "must be %q or %q, got %q", BuildNightly, BuildRelease, b.Source)
		}
		if strings.TrimSpace(b.Window) == "" {
			p.Field(at+".window", "required: how the commits were found, so a reader can redo it from the router's history")
		}
		if b.Source == BuildRelease && len(b.Commits) > 1 {
			p.Field(at+".commits", "a release is the one commit of its tag, got %d", len(b.Commits))
		}
		if len(b.Commits) == 0 {
			p.Field(at+".commits", "required: every commit the build could have been made from")
		}
		for j, c := range b.Commits {
			cat := fmt.Sprintf("%s.commits[%d]", at, j)
			switch {
			case !commit.MatchString(c):
				p.Field(cat, "must be a full 40-character lowercase commit, got %q", c)
			case j > 0 && c < b.Commits[j-1]:
				p.Field(cat, "commits are sorted")
			case j > 0 && c == b.Commits[j-1]:
				p.Field(cat, "repeats %s", c)
			}
			covered[c] = true
		}
		files := map[string]bool{}
		if len(b.Files) == 0 {
			p.Field(at+".files", "required: the files that decide the mode, each with the blob identifier it has at every commit of the build")
		}
		for j, f := range b.Files {
			fat := fmt.Sprintf("%s.files[%d]", at, j)
			switch {
			case f.Path == "" || strings.HasPrefix(f.Path, "/") || strings.Contains(f.Path, ".."):
				p.Field(fat+".path", "is a path from the repository's root, got %q", f.Path)
			case files[f.Path]:
				p.Field(fat+".path", "repeats %s", f.Path)
			}
			files[f.Path] = true
			if !commit.MatchString(f.Blob) {
				p.Field(fat+".blob", "must be a 40-character lowercase git blob identifier, got %q", f.Blob)
			}
			validateVariants(p, fat, f)
		}
		// A citation inside the window is of a file the build vouches for.
		in := map[string]bool{}
		for _, c := range b.Commits {
			in[c] = true
		}
		for k, c := range e.Evidence {
			if c.Side == SideRouter && in[c.Commit] && !files[c.Path] {
				p.Field(fmt.Sprintf("evidence[%d].path", k), "%s is not among the files of build %q, so its bytes at %s are not known to be those at the build's other commits", c.Path, b.Name, c.Commit)
			}
		}
	}
	// A commit in two builds would let one citation speak for both windows.
	seen := map[string]string{}
	for _, b := range e.Builds {
		for _, c := range b.Commits {
			if o, ok := seen[c]; ok && o != b.Name {
				p.Field("builds", "commit %s is in builds %q and %q: a window of one build is not another's", c, o, b.Name)
			}
			seen[c] = b.Name
		}
	}
	return covered
}

// ruleFor is the rule set that compares a deciding file's blobs: the Python rule for a Python
// file, the Rust rule for any other.
func ruleFor(path string) string {
	if strings.HasSuffix(path, ".py") {
		return NormalFormPythonRouting
	}
	return NormalFormLoggingOnly
}

var sha256hex = regexp.MustCompile(`^[0-9a-f]{64}$`)

// validateVariants checks a deciding file that has more than one blob in the window: the rule is
// the known one, every blob is distinct, and every variant has the normal form of the first blob.
// The data cannot be recomputed here (the router's source is third-party text and is not kept),
// so the check is that the claim is complete and consistent; the script that produced it is run
// on a clone of the router to redo it.
func validateVariants(p *validate.Problems, at string, f BuildFile) {
	if len(f.Variants) == 0 {
		if f.Rule != "" || f.NormalForm != "" {
			p.Field(at+".variants", "rule and normal_form are for a file with other blobs in the window; list them or drop both")
		}
		return
	}
	if want := ruleFor(f.Path); f.Rule != want {
		p.Field(at+".rule", "must be %q when the file has other blobs, got %q", want, f.Rule)
	}
	if !sha256hex.MatchString(f.NormalForm) {
		p.Field(at+".normal_form", "must be the 64-digit lowercase SHA-256 of the blob's normal form, got %q", f.NormalForm)
	}
	seen := map[string]bool{f.Blob: true}
	for i, v := range f.Variants {
		vat := fmt.Sprintf("%s.variants[%d]", at, i)
		switch {
		case !commit.MatchString(v.Blob):
			p.Field(vat+".blob", "must be a 40-character lowercase git blob identifier, got %q", v.Blob)
		case seen[v.Blob]:
			p.Field(vat+".blob", "repeats blob %s", v.Blob)
		}
		seen[v.Blob] = true
		if v.NormalForm != f.NormalForm {
			p.Field(vat+".normal_form", "%s has a different normal form from blob %s, so the file differs in more than %s removes", v.Blob, f.Blob, f.Rule)
		}
	}
}
