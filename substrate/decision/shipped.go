package decision

import (
	"embed"
	"io/fs"
)

// The artifact of this release: artifact.json, the simulator model and the fallback model of each
// front. It is data (design §3.3), embedded so that blis front works without a path, and a candidate
// artifact in a directory replaces it (LoadStage).
//
//go:embed artifact
var shippedArtifact embed.FS

// LoadShipped reads the artifact embedded in this build, under the same checks as LoadStage.
func LoadShipped(epsilon float64) (*Stage, *Artifact, error) {
	sub, err := fs.Sub(shippedArtifact, "artifact")
	if err != nil {
		return nil, nil, err
	}
	return loadStageFS(sub, "the shipped artifact", epsilon)
}
