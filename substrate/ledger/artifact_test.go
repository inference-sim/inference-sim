package ledger

import (
	"strings"
	"testing"
)

// D3 for a workflow artifact: an artifact is named by run id and attempt, its commit is full,
// and its zip and files are pinned by SHA-256. Each known-bad entry is refused.
func TestWorkflowArtifactPinIsExact(t *testing.T) {
	good := func() *Sources {
		s := goodSources()
		s.Sources[0].Artifacts = []Artifact{{
			ID: "e2e-accuracy-web", UsedBy: "the extractor",
			Revision: Revision{
				Kind: "workflow_artifact", URL: "https://github.com/o/r/actions/runs/36849388864",
				Ref: "36849388864", RunAttempt: 1, Commit: strings.Repeat("a", 40),
				Artifact: "e2e-accuracy-web-0d6e4079e36703eb",
				Assets: []Asset{
					{Name: "e2e-accuracy-web-0d6e4079e36703eb.zip", SHA256: strings.Repeat("b", 64)},
					{Name: "summary.json", SHA256: strings.Repeat("c", 64)},
				},
			},
		}}
		return s
	}
	if err := good().Validate(); err != nil {
		t.Fatalf("a pinned artifact was refused: %v", err)
	}
	for name, c := range map[string]struct {
		mut  func(a *Artifact)
		want string
	}{
		"run id latest":      {func(a *Artifact) { a.Revision.Ref = "latest" }, "not a workflow run id"},
		"no run id":          {func(a *Artifact) { a.Revision.Ref = "" }, "not a workflow run id"},
		"no attempt":         {func(a *Artifact) { a.Revision.RunAttempt = 0 }, "name the attempt"},
		"a branch":           {func(a *Artifact) { a.Revision.Commit = "main" }, "40-digit commit"},
		"a short commit":     {func(a *Artifact) { a.Revision.Commit = "cab77b9" }, "40-digit commit"},
		"artifact latest":    {func(a *Artifact) { a.Revision.Artifact = "latest" }, "not a pinned revision"},
		"no zip digest":      {func(a *Artifact) { a.Revision.Assets[0].Name = "other.zip" }, "is not pinned by sha256"},
		"a short digest":     {func(a *Artifact) { a.Revision.Assets[1].SHA256 = "abc" }, "64-digit sha256"},
		"url of another run": {func(a *Artifact) { a.Revision.URL = "https://github.com/o/r/actions/runs/1" }, "not the page of run"},
		"wrong kind":         {func(a *Artifact) { a.Revision.Kind = "git_commit" }, "want workflow_artifact"},
		"no tool named":      {func(a *Artifact) { a.UsedBy = "" }, "used_by"},
	} {
		s := good()
		c.mut(&s.Sources[0].Artifacts[0])
		if err := s.Validate(); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error containing %q", name, err, c.want)
		}
	}
}
