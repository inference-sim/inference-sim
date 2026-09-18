package latency

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// commentBlockJoiner matches a newline plus the `//` prefix that continues a comment block, so
// replacing it with a space flows a wrapped comment into one logical line.
var commentBlockJoiner = regexp.MustCompile(`\n[\t ]*//[\t ]?`)

// TestKDAScopeCommentDoesNotRetireLandedWork guards the KVBearingLayers scope comment in
// config.go (#1775).
//
// The retired claim: "KDA weights (#1638) and KDA step time (#1636) are out of scope and
// still use all NumLayers." #1636 landed — both roofline (roofline.go, via
// EffectiveKVBearingLayers) and trained-physics (trained_physics_model.go, via
// numKVBearingLayers) charge the sequence-length-dependent attention cost over the
// full-attention layers only and price the remaining KDA layers as linear attention. The
// comment sat directly above the field those backends read, so a contributor trusting it
// would conclude the work was still to do.
//
// This is a source-text guard on a COMMENT, which is weaker than a behavioral test — the
// behavior itself is covered by kda_step_time_test.go, and this test does not restate it.
// What it adds is the one thing a behavioral test cannot: a comment can go stale while every
// behavioral test still passes, which is exactly how this falsehood survived #1636. So the
// pattern below is deliberately narrow: it matches the FALSEHOOD (an out-of-scope claim
// naming #1636), not a wording. Rewording the comment freely is fine; re-asserting that
// #1636 is unimplemented is not.
func TestKDAScopeCommentDoesNotRetireLandedWork(t *testing.T) {
	src, err := os.ReadFile("config.go")
	if err != nil {
		t.Fatalf("read config.go: %v", err)
	}
	text := string(src)

	// Non-vacuity: the comment block this guards must still exist. If the field is renamed
	// or the comment removed, fail loudly rather than pass on an empty scan.
	if !strings.Contains(text, "kvBearingLayers := hf.LinearAttnFullLayerCount()") {
		t.Fatal("config.go no longer derives kvBearingLayers from LinearAttnFullLayerCount " +
			"(renamed? update this test and re-check the surrounding scope comment)")
	}
	if !strings.Contains(text, "#1636") {
		t.Error("the KVBearingLayers comment in config.go should still cite #1636 — it is what makes " +
			"the field a step-time input as well as a KV-capacity one")
	}

	// Match against FLOWED comment text, not raw source: a comment block is joined into one
	// logical line so a claim is found wherever the 100-column wrap happens to fall. Matching
	// raw source instead would make the test depend on line breaks — it would miss a claim
	// split across two lines, and would spuriously fail when a reflow moved one.
	flowed := commentBlockJoiner.ReplaceAllString(text, " ")

	// The falsehood's assertion form: #1636 named as out of scope / unimplemented. Bounded so
	// the two halves must sit in the same comment block, and direction-agnostic so both
	// "#1636 is out of scope" and "out of scope: #1636" match.
	staleClaims := []*regexp.Regexp{
		regexp.MustCompile(`(?i)#1636[^\n]{0,160}(out of scope|not (?:yet )?(?:modeled|modelled|implemented))`),
		regexp.MustCompile(`(?i)(out of scope|not (?:yet )?(?:modeled|modelled|implemented))[^\n]{0,160}#1636`),
	}
	for _, re := range staleClaims {
		if loc := re.FindStringIndex(flowed); loc != nil {
			t.Errorf("config.go describes #1636 (hybrid KDA step time) as unimplemented (%q), but both "+
				"latency backends model it — see EffectiveKVBearingLayers in roofline.go and "+
				"numKVBearingLayers in trained_physics_model.go, and the behavior tests in "+
				"kda_step_time_test.go. Only KDA *weights* (#1638) remain out of scope.",
				flowed[loc[0]:loc[1]])
		}
	}

	// #1638 (KDA weights) genuinely IS still out of scope, so the comment must keep saying so
	// — otherwise this correction would trade one falsehood for its opposite.
	if !regexp.MustCompile(`(?i)#1638[^\n]{0,160}out of scope`).MatchString(flowed) {
		t.Error("config.go must still record that KDA weights (#1638) are out of scope and charged as " +
			"full attention over all NumLayers; dropping that caveat overstates what #1636 delivered")
	}
}
