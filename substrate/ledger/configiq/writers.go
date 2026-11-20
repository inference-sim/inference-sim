package configiq

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Three writers feed the private CSV that the dataset is a filtered copy of: forge (Red Hat's
// open-source benchmark launcher), manual imports, and an earlier private launcher. The dataset
// records no writer, so a unit's family is read from its identifying fields, and the family
// decides how a blank is read and which blanks may be filled (M3.2 requirements §3C.1 to
// §3C.3, evidence/CONFIGIQ-FINDINGS.md). The rules are proposals pending the configiq team's
// answers; an answer that contradicts one replaces it.

// Family is the writer of a unit's row, recognised by the request type.
type Family string

const (
	// FamilyA is forge's: a /v1/... request type and prefix caching on.
	FamilyA Family = "A"
	// FamilyC is manual imports: the request type chat_completions or text_completions.
	FamilyC Family = "C"
	// FamilyD is the earlier launcher and older manual imports: a blank request type.
	FamilyD Family = "D"
)

// BlankReading is what a blank engine column means in a family's rows.
type BlankReading string

const (
	// BlankNotPassed: the writer passes its arguments as `key: value` pairs, so the publisher's
	// pattern sees every flag it set, and a blank is a flag that was not passed.
	BlankNotPassed BlankReading = "not_passed"
	// BlankUnknown: the pattern misses space-separated values, bare flags and names with
	// underscores, so a blank can hide a value the run set. It is never read as "on".
	BlankUnknown BlankReading = "unknown"
)

// Fill is one named substitution for a value the dataset does not record.
type Fill struct {
	Field string
	Value string
	// OnlyIfBlank: the fill applies only where the dataset's column is blank; a recorded value
	// is kept.
	OnlyIfBlank bool
	// Estimated: the value is inferred, not read from the writer's own records, so the point is
	// graded lower (§3C.3).
	Estimated bool
	// Source is where the value comes from: the writer's records or the evidence file.
	Source string
}

// WriterRule is the rule for one family.
type WriterRule struct {
	Family    Family
	Writer    string
	Signature string
	Blank     BlankReading
	// Fills are the substitutions the family's points carry, each named as such.
	Fills []Fill
	// UnscoredAtTP1: a row whose tensor-parallel size is 1 may have used data parallelism,
	// which the dataset does not record, so it stays unscored.
	UnscoredAtTP1 bool
	// ScoreOrder is the order the families are scored in: the lowest first.
	ScoreOrder int
}

const (
	srcForge  = "forge (openshift-psap/forge, checked at 673b4dfa): cpt.d/cpt.yaml, config.d/models.yaml and config.d/rhaiis.yaml; evidence/CONFIGIQ-FINDINGS.md"
	srcVLLM   = "vLLM's defaults at the run's release and device; evidence/CONFIGIQ-FINDINGS.md"
	srcFrame  = "forge's history: GuideLLM profile, duration, warm-up and ramp-up, seed and flags by commit; evidence/CONFIGIQ-FINDINGS.md"
	srcGuide  = "LLM-inference-benchmark-guide and the earlier launcher's defaults; evidence/CONFIGIQ-FINDINGS.md (inferred, medium confidence)"
	srcLittle = "Little's law on each concurrency-1 row; evidence/CONFIGIQ-FINDINGS.md"
)

var rules = []WriterRule{
	{
		Family:    FamilyA,
		Writer:    "forge's CPT",
		Signature: "request_type starts /v1/ and prefix_caching is yes",
		Blank:     BlankNotPassed,
		Fills: []Fill{
			{Field: "launch settings", Value: "the CPT matrix and presets at the commit in force on the run's date, matched on model, TP, KV dtype and request type", Source: srcForge},
			{Field: "max_num_seqs, chunked prefill, max_num_batched_tokens, precision", Value: "vLLM's default for the release and device", Source: srcVLLM},
			{Field: "frame", Value: "concurrent profile; duration, warm-up and ramp-up in force (275 s from 2026-09-22); GuideLLM release; seed 42; ignore_eos; streaming", Source: srcFrame},
		},
		ScoreOrder: 1,
	},
	{
		Family:        FamilyC,
		Writer:        "manual imports",
		Signature:     "request_type is chat_completions or text_completions",
		Blank:         BlankUnknown,
		UnscoredAtTP1: true,
		ScoreOrder:    2,
	},
	{
		Family:    FamilyD,
		Writer:    "the earlier launcher and older manual imports",
		Signature: "request_type is blank",
		Blank:     BlankUnknown,
		Fills: []Fill{
			{Field: "prefix_caching", Value: "off", OnlyIfBlank: true, Estimated: true, Source: srcGuide},
			{Field: "measurement duration", Value: "450 s; 600 s for vLLM 0.13 and older and for 8k/1k workloads on vLLM 0.20 and older", Estimated: true, Source: srcLittle},
		},
		UnscoredAtTP1: true,
		ScoreOrder:    2,
	},
}

// Rules returns the rule of each family: A, C and D. No family B was found.
func Rules() []WriterRule {
	out := make([]WriterRule, len(rules))
	copy(out, rules)
	return out
}

// RuleFor returns the rule of a family, or nil.
func RuleFor(f Family) *WriterRule {
	for i := range rules {
		if rules[i].Family == f {
			r := rules[i]
			return &r
		}
	}
	return nil
}

func identityString(identity map[string]any, key string) (string, error) {
	switch v := identity[key].(type) {
	case nil:
		return "", nil
	case string:
		return v, nil
	default:
		return "", fmt.Errorf("identity.%s: a %T, want a string or null", key, v)
	}
}

func identityInt(identity map[string]any, key string) (int64, bool) {
	switch v := identity[key].(type) {
	case int:
		return int64(v), true
	case int64:
		return v, true
	case json.Number:
		n, err := v.Int64()
		return n, err == nil
	}
	return 0, false
}

// WriterFamily classifies a configiq identity (a unit's, or Row.Identity's) by its writer. A
// row no family names is refused, not guessed: it is a writer the rules do not know.
func WriterFamily(identity map[string]any) (Family, error) {
	rt, err := identityString(identity, "request_type")
	if err != nil {
		return "", err
	}
	pc, err := identityString(identity, "prefix_caching")
	if err != nil {
		return "", err
	}
	switch {
	case strings.HasPrefix(rt, "/v1/"):
		if pc != "yes" {
			return "", fmt.Errorf("request_type %q with prefix_caching %q: forge's rows have prefix caching on, so this row belongs to no family", rt, pc)
		}
		return FamilyA, nil
	case rt == "chat_completions" || rt == "text_completions":
		return FamilyC, nil
	case rt == "":
		return FamilyD, nil
	}
	return "", fmt.Errorf("request_type %q belongs to no family", rt)
}

// UnscoredReason returns why a unit stays unscored under its family's rule, or "".
func UnscoredReason(identity map[string]any) (string, error) {
	f, err := WriterFamily(identity)
	if err != nil {
		return "", err
	}
	if !RuleFor(f).UnscoredAtTP1 {
		return "", nil
	}
	if tp, ok := identityInt(identity, "tp"); ok && tp == 1 {
		return "tp_1_may_use_data_parallelism", nil
	}
	return "", nil
}

// Substitutions returns the fills a unit's point carries under its family's rule. Family D's
// apply only to the earlier launcher's usual settings, a max_seq_len of 8192 or 10000 and a
// memory utilisation of 0.92; a D unit outside them has no fill, and a prefix_caching the
// dataset records is kept.
func Substitutions(identity map[string]any) ([]Fill, error) {
	f, err := WriterFamily(identity)
	if err != nil {
		return nil, err
	}
	fills := RuleFor(f).Fills
	if f != FamilyD {
		return append([]Fill(nil), fills...), nil
	}
	msl, err := identityString(identity, "max_seq_len")
	if err != nil {
		return nil, err
	}
	gmu, err := identityString(identity, "gpu_memory_utilization")
	if err != nil {
		return nil, err
	}
	if (msl != "8192" && msl != "10000") || gmu != "0.92" {
		return nil, nil
	}
	pc, err := identityString(identity, "prefix_caching")
	if err != nil {
		return nil, err
	}
	var out []Fill
	for _, fl := range fills {
		if fl.OnlyIfBlank && pc != "" {
			continue
		}
		out = append(out, fl)
	}
	return out, nil
}
