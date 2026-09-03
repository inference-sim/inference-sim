// interconnect_calib_file_test.go — companion invariant over the COMMITTED
// hardware_config.json for the interconnect calibration (#1530).
//
// The bundled calibration values are golden data. A companion invariant is required (R7)
// because a golden value only says "the file did not change", not "the file is correct" —
// and here a wrong file is worse than usual: a half-set bandwidth pair makes every
// trained-physics run on that GPU fail at startup, and an implausible ratio silently
// dominates step time for any spanning placement. These checks enumerate the file's own
// keys rather than a hardcoded list, so a newly added GPU is covered automatically.
package latency_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim/latency"
)

// committedGPUNames returns every GPU key in the repo's hardware_config.json, so these
// invariants cannot silently skip an entry someone adds later.
func committedGPUNames(t *testing.T) (string, []string) {
	t.Helper()
	path := filepath.Join("..", "..", "hardware_config.json")
	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	var entries map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &entries))
	require.NotEmpty(t, entries)

	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	return path, names
}

// TestCommittedHardwareConfig_InterconnectIsLoadable verifies that every committed GPU
// entry passes the interconnect validation. Without this, a half-set bandwidth pair in
// the bundled file would turn into a startup failure for every trained-physics run on
// that GPU — discovered by a user, not by CI.
func TestCommittedHardwareConfig_InterconnectIsLoadable(t *testing.T) {
	path, names := committedGPUNames(t)
	for _, gpu := range names {
		t.Run(gpu, func(t *testing.T) {
			_, err := latency.GetHWConfig(path, gpu)
			assert.NoError(t, err, "committed calibration for %q must load (GetHWConfig validates the interconnect)", gpu)
		})
	}
}

// TestCommittedHardwareConfig_InterconnectIsComplete verifies that every committed GPU
// declares BOTH interconnect bandwidths. They are optional in the schema — omitting both
// is valid and simply prices cross-node traffic at the on-node rate — but the bundled
// file is what most users run, so an entry missing them would leave multi-node estimates
// silently optimistic for that GPU.
func TestCommittedHardwareConfig_InterconnectIsComplete(t *testing.T) {
	path, names := committedGPUNames(t)
	for _, gpu := range names {
		t.Run(gpu, func(t *testing.T) {
			hc, err := latency.GetHWConfig(path, gpu)
			require.NoError(t, err)
			assert.Greater(t, hc.IntraNodeBwGBps, 0.0,
				"GPU %q must declare IntraNodeBwGBps so a spanning placement is priced (#1530)", gpu)
			assert.Greater(t, hc.InterNodeBwGBps, 0.0,
				"GPU %q must declare InterNodeBwGBps so a spanning placement is priced (#1530)", gpu)
		})
	}
}

// TestCommittedHardwareConfig_InterconnectRatioIsPlausible verifies the committed ratios
// sit in the range real hardware occupies. This is the check that would catch a
// bits-vs-bytes slip or a mistyped exponent — the failure mode the cost model is most
// sensitive to, since the cross-node penalty is linear in the ratio.
//
// Bounds: an inter-node fabric is never FASTER than the on-node link (ratio >= 1), and
// the widest real gap is roughly NVLink against a single 100 GbE uplink shared by a whole
// node (~75x). 100x is a deliberately loose ceiling.
func TestCommittedHardwareConfig_InterconnectRatioIsPlausible(t *testing.T) {
	path, names := committedGPUNames(t)
	for _, gpu := range names {
		t.Run(gpu, func(t *testing.T) {
			hc, err := latency.GetHWConfig(path, gpu)
			require.NoError(t, err)
			ratio := hc.InterconnectBwRatio()
			assert.GreaterOrEqual(t, ratio, 1.0,
				"GPU %q: an inter-node fabric cannot be faster than the on-node link (got %.2fx)", gpu, ratio)
			assert.LessOrEqual(t, ratio, 100.0,
				"GPU %q: ratio %.2fx is outside the range real hardware occupies — check for a "+
					"bits-vs-bytes or exponent error (both fields are per-GPU GB/s)", gpu, ratio)
		})
	}
}

// TestCommittedHardwareConfig_LatencyIsUncalibrated pins the deliberate decision that the
// bundled file declares NO per-collective inter-node latency. That term is the
// size-independent half of the cross-node cost and can exceed the bandwidth half for
// small decode messages, but BLIS has no measured value to ship, and shipping a guessed
// one would put a fabricated constant in front of every multi-node estimate (#1661).
//
// If a calibrated value is ever added, this test should be replaced by a plausibility
// range — not deleted — so the number stays under review.
func TestCommittedHardwareConfig_LatencyIsUncalibrated(t *testing.T) {
	path, names := committedGPUNames(t)
	for _, gpu := range names {
		hc, err := latency.GetHWConfig(path, gpu)
		require.NoError(t, err)
		assert.Zero(t, hc.EffectiveInterNodeLatencyUs(),
			"GPU %q declares a per-collective inter-node latency. That is a real fidelity "+
				"improvement, but it must come with a measured source (#1661) — update this test with the "+
				"plausibility range rather than removing it", gpu)
	}
}
