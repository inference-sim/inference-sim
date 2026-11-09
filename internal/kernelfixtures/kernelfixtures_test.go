package kernelfixtures

import (
	"os"
	"path/filepath"
	"testing"
)

// Each directory exists and holds fixtures; a wrong root would otherwise surface far away, as a
// kernel that cannot open its scenario.
func TestTheFixtureDirectoriesHoldScenarios(t *testing.T) {
	for _, dir := range []string{AISimulate(), Direct()} {
		m, err := filepath.Glob(filepath.Join(dir, "*.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		if len(m) < 50 {
			t.Errorf("%s holds %d fixtures; want at least 50", dir, len(m))
		}
	}
	if _, err := os.Stat(filepath.Join(Root(), "glm5-h200-tp8.yaml")); err != nil {
		t.Errorf("hand-written fixture missing: %v", err)
	}
}
