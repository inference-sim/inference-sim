package harness

import (
	"testing"

	"github.com/inference-sim/inference-sim/tools/harness/internal/artifacts"
)

func TestMain(m *testing.M) { artifacts.Main(m) }
