package harness

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel/internal/artifacts"
)

func TestMain(m *testing.M) { artifacts.Main(m) }
