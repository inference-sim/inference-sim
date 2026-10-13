package cmd

import (
	"fmt"
	"strings"
	"testing"

	schemaworkload "github.com/inference-sim/blis-schemas/spec/workload"
	"pgregory.net/rapid"
)

// Properties of the two catalog readers that moved onto blis-schemas' loaders. Each is a law
// over generated inputs rather than a value at one fixture, so it holds however the shipped
// presets and device classes change.

// tokenCount draws a token count across the interesting region: below the floor of 1, the
// small values bounds checks hinge on, and a long tail.
func tokenCount(t *rapid.T, label string) int {
	return rapid.OneOf(
		rapid.IntRange(-2, 4),
		rapid.IntRange(1, 64),
		rapid.IntRange(1, 1<<17),
	).Draw(t, label)
}

// A catalog preset is accepted exactly when BOTH owners of its rules accept it: the CLI
// flags' bound validation (R23 -- the same numbers as --prompt-tokens-* / --output-tokens-*
// are refused identically) and blis-schemas' own validation of the workload format. And an
// accepted preset reaches the simulator with every number it stated. A reader that dropped,
// renamed or re-defaulted a field would break the round trip; one that skipped either owner's
// rules, or added one of its own, would break the acceptance law.
//
// The schema's Validate is the oracle for the schema half rather than a restatement of its
// rules, so a rule blis-schemas adds is picked up here without an edit. (This property is what
// found that the schema refuses a prefix longer than the mean prompt, which the flags do not.)
func TestPresetReader_AcceptsWhatTheFlagsAndTheSchemaAcceptAndLosesNothing(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		want := presetWorkload{
			PrefixTokens:      rapid.IntRange(0, 4096).Draw(rt, "prefix"),
			PromptTokensMean:  tokenCount(rt, "promptMean"),
			PromptTokensStdev: tokenCount(rt, "promptStdev"),
			PromptTokensMin:   tokenCount(rt, "promptMin"),
			PromptTokensMax:   tokenCount(rt, "promptMax"),
			OutputTokensMean:  tokenCount(rt, "outputMean"),
			OutputTokensStdev: tokenCount(rt, "outputStdev"),
			OutputTokensMin:   tokenCount(rt, "outputMin"),
			OutputTokensMax:   tokenCount(rt, "outputMax"),
		}
		catalog := writeTestPresetCatalog(t, map[string]string{"chatbot": presetYAML(want)})
		got, err := readCatalogPresetWorkload("chatbot", catalog)

		flagsAccept := want.PromptTokensMean > 0 && want.OutputTokensMean > 0 &&
			validateDistributionParams(
				want.PromptTokensMin, want.PromptTokensMax,
				want.OutputTokensMin, want.OutputTokensMax,
				want.PromptTokensStdev, want.OutputTokensStdev,
				want.PromptTokensMean, want.OutputTokensMean) == ""
		shape := schemaworkload.Shape{
			Name:         "chatbot",
			PrefixTokens: want.PrefixTokens,
			Prompt: schemaworkload.Distribution{Mean: want.PromptTokensMean,
				StdDev: want.PromptTokensStdev, Min: want.PromptTokensMin, Max: want.PromptTokensMax},
			Output: schemaworkload.Distribution{Mean: want.OutputTokensMean,
				StdDev: want.OutputTokensStdev, Min: want.OutputTokensMin, Max: want.OutputTokensMax},
		}
		schemaAccepts := shape.Validate().OK()
		if accept := flagsAccept && schemaAccepts; accept != (err == nil) {
			rt.Fatalf("flags accept=%t, schema accepts=%t, but the catalog reader returned "+
				"err=%v for %+v", flagsAccept, schemaAccepts, err, want)
		}
		if err == nil && *got != want {
			rt.Fatalf("the preset did not round-trip:\n wrote %+v\n  read %+v", want, *got)
		}
	})
}

// A storage tier's figures reach the offload model unchanged: the catalog's _mb_s and _us
// keys are the same numbers as BLIS's bytes/µs and µs (1 MB/s = 1 byte/µs), so any
// conversion would be a bug. And a non-positive figure is refused naming its key -- a zero
// base latency makes a small transfer free, which no device is.
func TestStorageDeviceReader_PassesFiguresThroughAndRefusesNonPositive(t *testing.T) {
	keys := []string{"read_bandwidth_mb_s", "write_bandwidth_mb_s", "base_latency_us"}
	rapid.Check(t, func(rt *rapid.T) {
		figures := make([]float64, len(keys))
		bad := -1
		for i, k := range keys {
			figures[i] = rapid.Float64Range(1e-3, 1e6).Draw(rt, k)
		}
		if rapid.Bool().Draw(rt, "corrupt") {
			bad = rapid.IntRange(0, len(keys)-1).Draw(rt, "which")
			figures[bad] = rapid.SampledFrom([]float64{0, -1, -1e6}).Draw(rt, "value")
		}
		fields := make([]string, len(keys))
		for i, k := range keys {
			fields[i] = fmt.Sprintf("%s: %v", k, figures[i])
		}
		catalog := writeCatalogStorageDevices(t, "tier: {"+strings.Join(fields, ", ")+"}\n")
		devices, err := loadCatalogStorageDevices(catalog)

		if bad >= 0 {
			if err == nil || !strings.Contains(err.Error(), keys[bad]) {
				rt.Fatalf("%s=%v must be refused naming the key, got err=%v",
					keys[bad], figures[bad], err)
			}
			return
		}
		if err != nil {
			rt.Fatalf("a positive table was refused: %v", err)
		}
		got := devices["tier"]
		if got.ReadBandwidth != figures[0] || got.WriteBandwidth != figures[1] ||
			got.BaseLatency != figures[2] {
			rt.Fatalf("figures changed in transit: wrote %v, read %+v", figures, got)
		}
	})
}
