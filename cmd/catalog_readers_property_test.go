package cmd

import (
	"fmt"
	"strings"
	"testing"

	schemaworkload "github.com/inference-sim/blis-substrate/contracts/spec/workload"
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
		// The preset reaches the reader through the loader, as a run's does: the loader's strict
		// decoding and the schema's validation first, then the reader's bounds.
		data, err := dataWithCatalogOverlay(writeTestPresetCatalog(t, map[string]string{"chatbot": presetYAML(want)}))
		var got *presetWorkload
		if err == nil {
			got, err = readCatalogPresetWorkload("chatbot", data)
		}

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

// A storage tier is a name in the catalog and states no number: its read rate, write rate and
// base latency are registry entries priced by the kernel. A table that still states one of the
// three is refused naming the key, whatever its value, and a table of names alone resolves each
// to a class.
func TestStorageDeviceReader_ResolvesNamesAndRefusesAStatedRate(t *testing.T) {
	keys := []string{"read_bandwidth_mb_s", "write_bandwidth_mb_s", "base_latency_us"}
	rapid.Check(t, func(rt *rapid.T) {
		if rapid.Bool().Draw(rt, "states a rate") {
			key := rapid.SampledFrom(keys).Draw(rt, "key")
			value := rapid.Float64Range(-1e6, 1e6).Draw(rt, "value")
			_, err := dataFromCatalog(writeCatalogStorageDevices(t, fmt.Sprintf("tier: {_comment: \"a class\", %s: %v}\n", key, value)))
			if err == nil || !strings.Contains(err.Error(), key) {
				rt.Fatalf("%s: %v must be refused naming the key, got err=%v", key, value, err)
			}
			return
		}
		data, err := dataFromCatalog(writeCatalogStorageDevices(t, "tier: {_comment: \"a class\"}\nother: {_comment: \"another\"}\n"))
		if err != nil {
			rt.Fatalf("a table of names was refused: %v", err)
		}
		devices, err := loadCatalogStorageDevices(data)
		if err != nil {
			rt.Fatalf("a table of names did not resolve: %v", err)
		}
		if _, ok := devices["tier"]; !ok || len(devices) != 2 {
			rt.Fatalf("resolved %v, want the classes tier and other", devices)
		}
	})
}
