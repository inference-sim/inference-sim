// tracev2_crossnode_test.go — the max_nodes_spanned header field (#1530).
//
// This field is the fence that stops a trace from a multi-node run being replayed as a
// single-node fleet (which would be measurably faster than the run that produced it). Two
// properties have to hold for the fence to work, and both are easy to break silently: the
// value must survive a write/read round-trip through the YAML tag, and it must be ABSENT
// from the header of a run that had no multi-node placement — otherwise every existing
// trace changes shape and the byte-identity guarantee goes with it.
package workload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeAndReload exports a header (with one record, so the CSV is valid) and reads it back.
func writeAndReload(t *testing.T, header *TraceHeader) (*TraceV2, string) {
	t.Helper()
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	dataPath := filepath.Join(dir, "data.csv")
	records := []TraceRecord{{RequestID: 0, ClientID: "c1", SLOClass: "standard",
		InputTokens: 16, OutputTokens: 8, Status: "ok"}}
	require.NoError(t, ExportTraceV2(header, records, headerPath, dataPath))

	loaded, err := LoadTraceV2(headerPath, dataPath)
	require.NoError(t, err)
	raw, err := os.ReadFile(headerPath)
	require.NoError(t, err)
	return loaded, string(raw)
}

// TestTraceHeader_MaxNodesSpanned_RoundTrips verifies the recorded span survives export
// and re-import. If the YAML tag were wrong the value would silently read back as 0 and
// the replay fence would never fire.
func TestTraceHeader_MaxNodesSpanned_RoundTrips(t *testing.T) {
	for _, span := range []int{2, 3, 8} {
		loaded, raw := writeAndReload(t, &TraceHeader{
			Version: 3, TimeUnit: "microseconds", Mode: "generated", MaxNodesSpanned: span,
		})
		assert.Equal(t, span, loaded.Header.MaxNodesSpanned, "recorded span must survive the round-trip")
		assert.Contains(t, raw, "max_nodes_spanned:", "the field must be written under its documented key")
	}
}

// TestTraceHeader_MaxNodesSpanned_OmittedWhenAbsent verifies the field vanishes from the
// serialized header at its zero value — the value `blis run` writes whenever there was no
// multi-node placement, so every trace BLIS wrote before this feature keeps exactly the
// bytes it had.
func TestTraceHeader_MaxNodesSpanned_OmittedWhenAbsent(t *testing.T) {
	loaded, raw := writeAndReload(t, &TraceHeader{
		Version: 3, TimeUnit: "microseconds", Mode: "generated",
	})
	assert.NotContains(t, raw, "max_nodes_spanned",
		"an unset span must not appear in the header at all")
	assert.Zero(t, loaded.Header.MaxNodesSpanned)
}

// TestTraceHeader_MaxNodesSpanned_OneIsNotOmittedBySerialization pins where the
// byte-identity guarantee actually comes from, because it is NOT `omitempty` alone.
// `omitempty` on an int drops only 0, so a span of exactly 1 — "placement happened, and
// every instance fit on one node" — WOULD be serialized. That is why the writer normalizes
// 1 to 0 before it reaches the header (cmd.crossNodeSpanForTrace, covered by its own test):
// without that normalization, every single-node node-pool run would start emitting a new
// header key.
//
// This test exists so that if someone ever removes the normalization believing `omitempty`
// covers it, the coupling is written down here rather than discovered as a changed trace
// format.
func TestTraceHeader_MaxNodesSpanned_OneIsNotOmittedBySerialization(t *testing.T) {
	_, raw := writeAndReload(t, &TraceHeader{
		Version: 3, TimeUnit: "microseconds", Mode: "generated", MaxNodesSpanned: 1,
	})
	assert.Contains(t, raw, "max_nodes_spanned",
		"a span of 1 is serialized, so the writer must normalize it to 0 rather than relying on omitempty")
}

// TestTraceHeader_MaxNodesSpanned_ZeroMatchesFieldlessHeader is the byte-identity check:
// a header whose span is unset must serialize to exactly what a header written before the
// field existed looks like.
func TestTraceHeader_MaxNodesSpanned_ZeroMatchesFieldlessHeader(t *testing.T) {
	_, withZero := writeAndReload(t, &TraceHeader{
		Version: 3, TimeUnit: "microseconds", Mode: "generated", WarmUpRequests: 0,
	})
	assert.False(t, strings.Contains(withZero, "max_nodes_spanned"),
		"the default header must carry no trace of the new field")
	// And it must still parse under strict loading, which the round-trip above exercised.
}

// TestTraceHeader_MaxNodesSpanned_AbsentKeyLoads verifies backward compatibility from the
// other direction: a header written before this field existed still loads, with the span
// reading back as 0 so the replay fence stays quiet.
func TestTraceHeader_MaxNodesSpanned_AbsentKeyLoads(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	dataPath := filepath.Join(dir, "data.csv")
	require.NoError(t, os.WriteFile(headerPath,
		[]byte("trace_version: 3\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"), 0644))
	require.NoError(t, os.WriteFile(dataPath, []byte(strings.Join(traceV2Columns, ",")+"\n"), 0644))

	loaded, err := LoadTraceV2(headerPath, dataPath)
	require.NoError(t, err, "a pre-feature header must still load")
	assert.Zero(t, loaded.Header.MaxNodesSpanned)
}
