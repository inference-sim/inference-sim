package routertable

import (
	"fmt"
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
