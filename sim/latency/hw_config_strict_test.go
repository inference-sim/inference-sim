// hw_config_strict_test.go — strict-parsing contracts for hardware_config.json (#1728).
//
// Before #1728 this was the only permissively-parsed config path in BLIS: a misspelled
// numeric key was dropped by the decoder and the field read 0, producing a
// plausible-but-wrong result (a 0 bandwidth, a 0 MFU, a 0 memory capacity) with no
// diagnostic anywhere. These tests fence the strict behavior AND the two things
// strictness must not break — the documentation-only provenance keys, and the
// case-insensitive field matching encoding/json performs.
package latency_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// hardwareCalibJSONKeys returns the canonical JSON key of every field declared on
// sim.HardwareCalib, read off the struct tags. Enumerating the struct rather than a
// hardcoded list is what makes the "every numeric field" contract (BC-2) cover fields
// added after this test was written.
func hardwareCalibJSONKeys(t *testing.T) []string {
	t.Helper()
	typ := reflect.TypeOf(sim.HardwareCalib{})
	keys := make([]string, 0, typ.NumField())
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		if !f.IsExported() {
			continue
		}
		name := f.Name
		if tag, ok := f.Tag.Lookup("json"); ok {
			if tagName := strings.Split(tag, ",")[0]; tagName == "-" {
				continue
			} else if tagName != "" {
				name = tagName
			}
		}
		keys = append(keys, name)
	}
	require.NotEmpty(t, keys)
	sort.Strings(keys)
	return keys
}

// writeHWConfig writes a one-GPU hardware config from raw key/value JSON fragments and
// returns its path.
func writeHWConfig(t *testing.T, gpu string, fields map[string]string) string {
	t.Helper()
	parts := make([]string, 0, len(fields))
	names := make([]string, 0, len(fields))
	for k := range fields {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		parts = append(parts, fmt.Sprintf("%q: %s", k, fields[k]))
	}
	body := fmt.Sprintf("{%q: {%s}}", gpu, strings.Join(parts, ", "))
	path := filepath.Join(t.TempDir(), "hardware_config.json")
	require.NoError(t, os.WriteFile(path, []byte(body), 0644))
	return path
}

// baseHWFields is a minimal VALID entry: enough for GetHWConfig to succeed, so a test
// that adds one bad key is isolating that key.
func baseHWFields() map[string]string {
	return map[string]string{
		"TFlopsPeak":      "989.5",
		"BwPeakTBs":       "3.35",
		"mfuPrefill":      "0.45",
		"mfuDecode":       "0.30",
		"MemoryGiB":       "80.0",
		"IntraNodeBwGBps": "450",
		"InterNodeBwGBps": "50",
	}
}

// TestStrictHWConfig_UnknownKeyIsRejected covers BC-1: an unknown key fails the load,
// and the error names both the key and the GPU entry it appears under (the operator has
// to be able to find it in a file with a dozen entries).
func TestStrictHWConfig_UnknownKeyIsRejected(t *testing.T) {
	fields := baseHWFields()
	delete(fields, "IntraNodeBwGBps")
	fields["IntraNodeBandwidthGBps"] = "450"

	path := writeHWConfig(t, "H100", fields)
	_, err := latency.GetHWConfig(path, "H100")
	require.Error(t, err, "an unknown key must be rejected, not silently read as 0")
	assert.Contains(t, err.Error(), "IntraNodeBandwidthGBps", "error must name the offending key")
	assert.Contains(t, err.Error(), "H100", "error must name the GPU entry the key appears under")
}

// TestStrictHWConfig_CaseMismatchIsRejectedNamingTheCanonicalKey covers BC-8 and the
// capitalization slip #1728 names as its motivating example (IntraNodeBwGbps for
// IntraNodeBwGBps).
//
// CORRECTION to the issue's premise: encoding/json falls back to a case-INSENSITIVE
// field match, so that particular slip loaded the value CORRECTLY before this change —
// it never produced the 0 bandwidth the issue describes (verified against
// encoding/json). The silent-zero class is keys that differ by more than case.
// A case slip is still rejected, since two spellings of one field in the same entry
// resolve last-wins and the file is otherwise one letter from a genuine typo — but the
// error has to say so, naming the canonical spelling, rather than calling it unknown.
func TestStrictHWConfig_CaseMismatchIsRejectedNamingTheCanonicalKey(t *testing.T) {
	fields := baseHWFields()
	delete(fields, "IntraNodeBwGBps")
	fields["IntraNodeBwGbps"] = "450"

	path := writeHWConfig(t, "H100", fields)
	_, err := latency.GetHWConfig(path, "H100")
	require.Error(t, err, "a key that differs from a declared field only in case must be rejected")
	assert.Contains(t, err.Error(), "IntraNodeBwGbps", "error must name the offending key")
	assert.Contains(t, err.Error(), "IntraNodeBwGBps", "error must name the canonical spelling to use")
	assert.Contains(t, err.Error(), "H100", "error must name the GPU entry the key appears under")
	assert.Contains(t, strings.ToLower(err.Error()), "case",
		"the diagnostic must say the problem is letter case, not an unknown field")
}

// TestStrictHWConfig_MisspelledNumericFieldIsRejected covers BC-2: EVERY numeric field
// declared on sim.HardwareCalib must fail when misspelled, rather than silently reading
// zero. The table is generated from the struct, so a field added later is covered
// automatically.
//
// The misspelling appends a character, so it can neither collide with another declared
// key nor case-fold onto one. That keeps every case in this table in the UNKNOWN-key
// class, which is the silent-zero class this test is about; a key that differs from a
// declared field only in case is also rejected, but as its own diagnostic class — see
// TestStrictHWConfig_CaseMismatchIsRejectedNamingTheCanonicalKey.
func TestStrictHWConfig_MisspelledNumericFieldIsRejected(t *testing.T) {
	for _, key := range hardwareCalibJSONKeys(t) {
		t.Run(key, func(t *testing.T) {
			fields := baseHWFields()
			delete(fields, key)
			typo := key + "Z"
			fields[typo] = "123.0"

			path := writeHWConfig(t, "H100", fields)
			_, err := latency.GetHWConfig(path, "H100")
			require.Error(t, err,
				"a misspelling of %q must be rejected — silently reading 0 yields a plausible-but-wrong result", key)
			assert.Contains(t, err.Error(), typo, "error must name the offending key")
			assert.Contains(t, err.Error(), "H100", "error must name the offending GPU entry")
		})
	}
}

// TestStrictHWConfig_CanonicalKeysAreAccepted covers BC-6: the accepted-key set is
// derived from sim.HardwareCalib, so the canonical key of every declared field loads.
// This is the companion of the misspelling table: without it, a parser that rejected
// everything would pass BC-2 vacuously.
func TestStrictHWConfig_CanonicalKeysAreAccepted(t *testing.T) {
	fields := baseHWFields()
	for _, key := range hardwareCalibJSONKeys(t) {
		if _, ok := fields[key]; !ok {
			// Fields not in the minimal valid entry (e.g. TFlopsFP8,
			// InterNodeHopLatencyUs) must still be accepted when declared.
			fields[key] = "1.0"
		}
	}
	path := writeHWConfig(t, "H100", fields)
	_, err := latency.GetHWConfig(path, "H100")
	require.NoError(t, err, "every key declared on sim.HardwareCalib must be accepted")
}

// TestStrictHWConfig_NoTwoFieldsDifferOnlyInCase guards the one way the accepted-key
// derivation could go wrong as sim.HardwareCalib grows: two json tags that are equal
// once case-folded would be ambiguous both to encoding/json's case-insensitive fallback
// and to the parser's case-mismatch diagnosis, so one of the two canonical spellings
// would be reported as a "case mismatch" for the other. Two fields must never differ
// only in letter case.
func TestStrictHWConfig_NoTwoFieldsDifferOnlyInCase(t *testing.T) {
	seen := map[string]string{}
	for _, key := range hardwareCalibJSONKeys(t) {
		folded := strings.ToLower(key)
		if prev, dup := seen[folded]; dup {
			t.Errorf("sim.HardwareCalib declares %q and %q, which differ only in letter case", prev, key)
		}
		seen[folded] = key
	}
}

// TestStrictHWConfig_ProvenanceKeysAreAccepted covers BC-3: the two documentation-only
// keys the bundled file carries per GPU entry survive strict parsing, and the numeric
// values around them are read correctly (i.e. they are ignored, not treated as data).
func TestStrictHWConfig_ProvenanceKeysAreAccepted(t *testing.T) {
	fields := baseHWFields()
	fields["_comment"] = `"MFU values calibrated per Discussion #589"`
	fields["_comment_interconnect"] = `"Per-GPU effective unidirectional GB/s; ratio 9x"`

	path := writeHWConfig(t, "H100", fields)
	hc, err := latency.GetHWConfig(path, "H100")
	require.NoError(t, err, "_comment / _comment_interconnect must still load (they carry calibration provenance)")
	assert.Equal(t, 989.5, hc.TFlopsPeak, "values alongside the provenance keys must be read normally")
	assert.Equal(t, 450.0, hc.IntraNodeBwGBps)
	assert.Equal(t, 50.0, hc.InterNodeBwGBps)
}

// TestStrictHWConfig_ProvenanceKeyCaseMismatchIsRejected checks that the provenance keys
// get the same treatment as the calibration fields: "_Comment" is diagnosed as a case
// mismatch naming "_comment", not accepted by accident and not called unknown. Without
// this, the two accepted-key classes would follow different rules for no reason.
func TestStrictHWConfig_ProvenanceKeyCaseMismatchIsRejected(t *testing.T) {
	fields := baseHWFields()
	fields["_Comment"] = `"calibrated per Discussion #589"`

	path := writeHWConfig(t, "H100", fields)
	_, err := latency.GetHWConfig(path, "H100")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "_comment", "error must name the canonical provenance key")
}

// TestStrictHWConfig_AllOffendersReportedDeterministically covers BC-7: several unknown
// keys across several GPUs are all reported, in a stable order, so the error text is
// reproducible run to run (INV-6) and the operator fixes the file in one pass instead of
// one key per run.
func TestStrictHWConfig_AllOffendersReportedDeterministically(t *testing.T) {
	body := `{
	  "H100": {"TFlopsPeak": 989.5, "BwPeakTBs": 3.35, "zzzUnknown": 1, "aaaUnknown": 2},
	  "A100-SXM": {"TFlopsPeak": 312, "BwPeakTBs": 2.039, "mfuTypo": 3}
	}`
	path := filepath.Join(t.TempDir(), "hardware_config.json")
	require.NoError(t, os.WriteFile(path, []byte(body), 0644))

	var first string
	for i := 0; i < 5; i++ {
		_, err := latency.GetHWConfig(path, "H100")
		require.Error(t, err)
		if i == 0 {
			first = err.Error()
			for _, want := range []string{"zzzUnknown", "aaaUnknown", "mfuTypo", "H100", "A100-SXM"} {
				assert.Contains(t, first, want, "every offender (and its GPU) must be reported")
			}
			// Sorted by GPU, then key: A100-SXM's offender precedes H100's, and
			// within H100 "aaaUnknown" precedes "zzzUnknown".
			assert.Less(t, strings.Index(first, "mfuTypo"), strings.Index(first, "aaaUnknown"),
				"offenders must be ordered by GPU name")
			assert.Less(t, strings.Index(first, "aaaUnknown"), strings.Index(first, "zzzUnknown"),
				"offenders within one GPU must be ordered by key")
			continue
		}
		assert.Equal(t, first, err.Error(), "the diagnostic must be byte-identical across repeated loads (INV-6)")
	}
}

// TestStrictHWConfig_CommittedFileDeclaresOnlyKnownKeys covers BC-4 over the committed
// hardware_config.json: every key in the shipped file is either a declared
// sim.HardwareCalib field or one of the two provenance keys. Without this, adding a GPU
// entry with a typo'd key would make the bundled file unloadable — a failure discovered
// by a user rather than by CI. It enumerates the file, so a newly added entry is covered.
func TestStrictHWConfig_CommittedFileDeclaresOnlyKnownKeys(t *testing.T) {
	path := filepath.Join("..", "..", "hardware_config.json")
	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	var entries map[string]map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &entries))
	require.NotEmpty(t, entries)

	known := map[string]bool{"_comment": true, "_comment_interconnect": true}
	for _, k := range hardwareCalibJSONKeys(t) {
		known[strings.ToLower(k)] = true
	}
	for gpu, fields := range entries {
		for key := range fields {
			assert.True(t, known[strings.ToLower(key)],
				"committed hardware config: GPU %q declares key %q, which strict parsing rejects", gpu, key)
		}
		// The whole entry must also load, i.e. strictness did not break the shipped file.
		_, err := latency.GetHWConfig(path, gpu)
		assert.NoError(t, err, "committed entry %q must still load unchanged (INV-6)", gpu)
	}
}
