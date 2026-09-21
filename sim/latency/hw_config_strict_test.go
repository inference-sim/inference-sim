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

// hardwareCalibField is one entry of sim.HardwareCalib's JSON surface: the canonical key
// the parser must accept, and the Go type a value under that key has to decode into.
// The type is carried so the acceptance tests can supply a value of the right SHAPE —
// see hardwareCalibJSONValue.
type hardwareCalibField struct {
	Key  string
	Type reflect.Type
}

// hardwareCalibJSONFields enumerates sim.HardwareCalib's JSON surface off the struct
// tags, mirroring what encoding/json exposes. Enumerating the struct rather than a
// hardcoded list is what makes the strictness contracts (BC-2, BC-6) cover fields added
// after these tests were written.
//
// It applies the SAME tag rules as the production key derivation
// (latency.buildHardwareCalibKnownKeys): skip unexported fields, honour a json tag name,
// skip `json:"-"`. Two independent implementations of one rule can drift, which is why
// TestStrictHWConfig_AcceptedKeySetMatchesTheReflectionPath asserts they agree, and why
// TestStrictHWConfig_NoEmbeddedFields fences the one rule NEITHER implements (#1777).
func hardwareCalibJSONFields(t *testing.T) []hardwareCalibField {
	t.Helper()
	typ := reflect.TypeOf(sim.HardwareCalib{})
	fields := make([]hardwareCalibField, 0, typ.NumField())
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
		fields = append(fields, hardwareCalibField{Key: name, Type: f.Type})
	}
	require.NotEmpty(t, fields)
	sort.Slice(fields, func(i, j int) bool { return fields[i].Key < fields[j].Key })
	return fields
}

// hardwareCalibJSONKeys returns the canonical JSON key of every field declared on
// sim.HardwareCalib, sorted.
func hardwareCalibJSONKeys(t *testing.T) []string {
	t.Helper()
	fields := hardwareCalibJSONFields(t)
	keys := make([]string, 0, len(fields))
	for _, f := range fields {
		keys = append(keys, f.Key)
	}
	return keys
}

// hardwareCalibJSONValue returns a valid JSON literal for a field of the given type, so
// an acceptance test can declare EVERY field — not only the float64 ones — with a value
// the decoder will take.
//
// Why it is type-driven rather than a fixed "1.0": every HardwareCalib field is a float64
// today, so a hardcoded number works today and would break the moment the struct gains a
// string, bool or nested field — turning an unrelated field addition into a confusing
// decode failure in a test about KEY strictness, and leaving the accepted-key contracts
// unproven for exactly the field that needed them (#1777 item 3). An unhandled kind is a
// t.Fatalf naming it, so a genuinely new shape gets a human decision instead of a value
// that happens to unmarshal.
func hardwareCalibJSONValue(t *testing.T, typ reflect.Type) string {
	t.Helper()
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	switch typ.Kind() {
	case reflect.Float32, reflect.Float64:
		return "123.0"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "123"
	case reflect.String:
		return `"x"`
	case reflect.Bool:
		return "true"
	case reflect.Slice, reflect.Array:
		return "[]"
	case reflect.Map, reflect.Struct:
		return "{}"
	default:
		t.Fatalf("sim.HardwareCalib declares a field of kind %s, which this test does not know how to "+
			"express as JSON. Teach hardwareCalibJSONValue about it (and check whether "+
			"buildHardwareCalibKnownKeys still derives the right key for it)", typ.Kind())
		return ""
	}
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
	for _, f := range hardwareCalibJSONFields(t) {
		key := f.Key
		t.Run(key, func(t *testing.T) {
			fields := baseHWFields()
			delete(fields, key)
			typo := key + "Z"
			fields[typo] = hardwareCalibJSONValue(t, f.Type)

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
	for _, f := range hardwareCalibJSONFields(t) {
		if _, ok := fields[f.Key]; !ok {
			// Fields not in the minimal valid entry (e.g. TFlopsFP8,
			// InterNodeHopLatencyUs) must still be accepted when declared. The value is
			// shaped from the field's type, so a future non-float64 field is covered here
			// rather than failing to decode (#1777 item 3).
			fields[f.Key] = hardwareCalibJSONValue(t, f.Type)
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

// TestStrictHWConfig_LegacyKeyWinsOverAnUnrelatedUnknownKey is #1777 item 2: the #1694
// migration guard must take precedence over the #1728 generic unknown-key diagnostic even
// when the generic scan has an INDEPENDENT reason to fire.
//
// Both guards run on every load, in a fixed order (rejectLegacyInterNodeLatencyKey, then
// rejectUnknownHardwareCalibKeys), and each returns the whole error. The existing coverage
// tests them in isolation: TestLegacyInterNodeLatencyKeyIsRejected supplies only the legacy
// key (where the generic scan would ALSO name it, since a removed key is an unknown key),
// and TestStrictHWConfig_UnknownKeyIsRejected supplies only a typo. Neither pins the case
// an operator mid-migration actually hits — an old calibrated value AND a fresh typo in the
// same file — where precedence decides which of two applicable messages they get.
//
// It has to be the migration message. The generic one advises deleting an unrecognized key;
// obeying it would silently discard a calibrated per-collective latency, when the correct
// action is to divide it by the hop count and set the per-HOP key (the unit changed in
// #1694, so it is a recalibration and not a rename). Being told about the typo first would
// send the operator down the wrong path with no hint that the hint existed.
func TestStrictHWConfig_LegacyKeyWinsOverAnUnrelatedUnknownKey(t *testing.T) {
	const unrelatedTypo = "MemoryGB" // the MemoryGiB slip #1728 names, unrelated to the migration

	fields := baseHWFields()
	fields["InterNodeLatencyUs"] = "542"
	fields[unrelatedTypo] = "80.0"

	path := writeHWConfig(t, "H100", fields)
	_, err := latency.GetHWConfig(path, "H100")
	require.Error(t, err, "a config with both a legacy key and a typo must be rejected")

	// The migration message, in full: the removed key, the replacement key, the unit-change
	// hint, and where to look.
	assert.Contains(t, err.Error(), "InterNodeLatencyUs", "error must name the removed key")
	assert.Contains(t, err.Error(), "InterNodeHopLatencyUs", "error must name the replacement key")
	assert.Contains(t, err.Error(), "RECALIBRATION",
		"the migration message must win: its unit-change hint is the whole reason the guard exists")
	assert.Contains(t, err.Error(), "H100", "error must name the offending GPU entry")

	// And NOT the generic one. Reporting both would bury the hint; reporting only the
	// generic one would lose it. The unrelated typo is not silently accepted — it is simply
	// not yet reported, and surfaces on the next load once the legacy key is removed (the
	// re-load below).
	assert.NotContains(t, err.Error(), "unrecognized key",
		"the generic unknown-key diagnostic must not preempt or accompany the migration message")
	assert.NotContains(t, err.Error(), unrelatedTypo,
		"the migration message must not be diluted with the unrelated typo")

	// Non-vacuity control: the same fixture MINUS the legacy key is still rejected, by the
	// generic guard, naming the typo. Without this the two NotContains assertions above
	// would also pass if MemoryGB were simply an accepted key.
	delete(fields, "InterNodeLatencyUs")
	path = writeHWConfig(t, "H100", fields)
	_, err = latency.GetHWConfig(path, "H100")
	require.Error(t, err, "the unrelated typo must be a genuine offender, else the precedence claim is vacuous")
	assert.Contains(t, err.Error(), unrelatedTypo,
		"once the legacy key is gone the generic diagnostic reports the typo")
	assert.Contains(t, err.Error(), "unrecognized key")
	assert.NotContains(t, err.Error(), "RECALIBRATION",
		"the migration message must not fire for a file that no longer carries the legacy key")
}

// TestStrictHWConfig_AcceptedKeySetMatchesTheReflectionPath is #1777 item 3: the set of
// keys the PARSER accepts and the set these tests derive from sim.HardwareCalib must be
// the same set.
//
// Two independent reflection walks exist — production's buildHardwareCalibKnownKeys and
// this file's hardwareCalibJSONFields — and every other strictness test in this file is
// written against the second one. If they disagree, those tests go quietly wrong in the
// worst direction: a field the parser rejects would be "covered" by an acceptance test
// that never exercises it, and a field the parser accepts but the helper misses would have
// no misspelling coverage at all. The other tests can only ever check keys the helper
// already knows about, so nothing else in this file can catch the divergence.
//
// The parser's set is read from its own diagnostic, which lists the valid keys — the only
// place production publishes it, and content an operator depends on, so asserting on it is
// asserting on behavior.
func TestStrictHWConfig_AcceptedKeySetMatchesTheReflectionPath(t *testing.T) {
	// Any unknown key elicits the advertised list.
	fields := baseHWFields()
	fields["definitelyNotAField"] = "1"
	path := writeHWConfig(t, "H100", fields)
	_, err := latency.GetHWConfig(path, "H100")
	require.Error(t, err)

	const marker = "Valid keys are ["
	start := strings.Index(err.Error(), marker)
	require.GreaterOrEqual(t, start, 0,
		"the unknown-key diagnostic must publish the valid-key list (it is how an operator finds the "+
			"right spelling); got: %v", err)
	rest := err.Error()[start+len(marker):]
	end := strings.Index(rest, "]")
	require.GreaterOrEqual(t, end, 0, "valid-key list must be bracketed; got: %v", err)
	advertised := strings.Fields(rest[:end])

	sort.Strings(advertised)
	assert.Equal(t, hardwareCalibJSONKeys(t), advertised,
		"the parser's accepted-key set and this file's derived set must agree — otherwise every other "+
			"strictness test in this file is measuring the wrong struct surface")
}

// TestStrictHWConfig_NoEmbeddedFields fences the one JSON rule NEITHER reflection walk
// implements (#1777 item 3): encoding/json PROMOTES an embedded struct's fields to the
// outer object, but both walks treat a field as one key named after itself, so an embedded
// field would be registered under the embedded TYPE's name — a key no config ever contains
// — while every field it actually promotes would be rejected as unknown.
//
// That is a hard failure at load for a valid config, and the failure mode is invisible in
// review: the struct change looks like a pure refactor, and every existing strictness test
// keeps passing because both walks are wrong in the same way (so the parity test above
// still sees two equal sets). Hence a direct structural guard.
//
// The fix, should an embedded field ever be wanted, is to teach both walks to recurse
// (honouring json:"name" on the embedded field itself, which SUPPRESSES promotion). It is
// deliberately not implemented today: HardwareCalib has no embedded field, so recursion
// would be untested code on a parser whose whole job is to fail closed.
func TestStrictHWConfig_NoEmbeddedFields(t *testing.T) {
	typ := reflect.TypeOf(sim.HardwareCalib{})
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		if f.Anonymous {
			t.Errorf("sim.HardwareCalib embeds %s. encoding/json promotes an embedded struct's fields to "+
				"the outer object, but the accepted-key derivation (both latency.buildHardwareCalibKnownKeys "+
				"and this file's helper) registers one key per declared field — so the promoted keys would be "+
				"rejected as unknown and a valid hardware config would fail to load. Teach both walks to "+
				"recurse into embedded structs, or give the field an explicit json tag to suppress promotion",
				f.Name)
		}
	}
}
