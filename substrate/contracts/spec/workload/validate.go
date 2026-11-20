package workload

import (
	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
)

// isHex reports whether every character of s is a hex digit. Used to reject a sha256 of
// the right length but the wrong alphabet — a digest that could never match a file.
func isHex(s string) bool {
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') && (c < 'A' || c > 'F') {
			return false
		}
	}
	return true
}

// Validate performs field-level validation of a workload binding: that exactly one arm
// is chosen, and that the chosen arm is itself well formed. A binding is validated only
// when present; a Scenario with no traffic carries no binding, which this type never
// sees (the Scenario skips a nil one).
func (b *Binding) Validate() *validate.Problems {
	p := &validate.Problems{}
	arms := 0
	for _, set := range []bool{b.Shape != "", b.Trace != nil, b.Replay != nil} {
		if set {
			arms++
		}
	}
	switch {
	case arms > 1:
		p.Errorf("a workload is one of a shape, a trace or a replay; set exactly one")
	case arms == 0:
		p.Errorf("a workload binding chooses a shape, a trace or a replay; omit the workload entirely for a scenario with no traffic")
	case b.Trace != nil:
		p.Merge("trace", b.Trace.Validate())
	case b.Replay != nil:
		p.Merge("replay", b.Replay.Validate())
	}
	return p
}

// Validate performs field-level validation of a trace reference: that it names a data
// file, that any integrity fields are well formed, and that the header is valid.
func (t *TraceRef) Validate() *validate.Problems {
	p := &validate.Problems{}
	if t.Data == "" {
		p.Field("data", "required: the path to the bulk per-request data CSV")
	}
	// Optional, but a digest that is present must be a plausible one: a wrong length or a
	// non-hex character is a copy-paste error that would never match the file it is meant
	// to guard, so it is better rejected here than discovered as a mismatch later.
	if t.SHA256 != "" {
		switch {
		case len(t.SHA256) != 64:
			p.Field("sha256",
				"must be a 64-character hex digest of the data file, got %d characters",
				len(t.SHA256))
		case !isHex(t.SHA256):
			p.Field("sha256", "must be a hex digest; it contains a non-hex character")
		}
	}
	if t.Rows < 0 {
		p.Field("rows", "must not be negative")
	}
	if b := t.TokenBlocks; b != nil {
		if b.Path == "" {
			p.Field("token_blocks.path", "required: the path to the block-hash sidecar")
		}
		p.SHA256Field("token_blocks.sha256", b.SHA256)
	}
	p.Merge("header", t.Header.Validate())
	acceptance.ValidateWorkloadTerms(p, t.Category, t.Thinking, t.Acceptance)
	if t.Routing != nil {
		p.Merge("routing", t.Routing.Validate())
	}
	return p
}

// Validate performs field-level validation of a traffic shape.
func (s *Shape) Validate() *validate.Problems {
	p := &validate.Problems{}
	// Name is required. It can no longer come from the file (the field is tagged
	// yaml:"-"), so the loader always supplies it and this check is unreachable through
	// LoadWorkload — but it still guards a Shape built directly in code or a
	// hand-populated Bundle.Workload, so it stays: a later cleanup must not read it as
	// dead.
	if s.Name == "" {
		p.Field("name", "required")
	}
	if s.PrefixTokens < 0 {
		p.Field("prefix_tokens", "must not be negative")
	}
	s.Prompt.validate(p, "prompt")
	s.Output.validate(p, "output")
	acceptance.ValidateWorkloadTerms(p, s.Category, s.Thinking, s.Acceptance)
	if s.Routing != nil {
		p.Merge("routing", s.Routing.Validate())
	}
	// A shared prefix longer than the prompt it is a prefix of describes no request.
	if s.Prompt.Mean > 0 && s.PrefixTokens > s.Prompt.Mean {
		p.Field("prefix_tokens",
			"%d exceeds the mean prompt length %d, so no prompt contains the prefix",
			s.PrefixTokens, s.Prompt.Mean)
	}
	return p
}

func (d Distribution) validate(p *validate.Problems, at string) {
	if d.Mean < 1 {
		p.Field(at+".tokens", "must be positive")
	}
	if d.StdDev < 0 {
		p.Field(at+".tokens_stdev", "must not be negative")
	}
	if d.Min < 0 {
		p.Field(at+".tokens_min", "must not be negative")
	}
	if d.Max > 0 && d.Min > d.Max {
		p.Field(at+".tokens_min", "%d exceeds tokens_max %d", d.Min, d.Max)
	}
	// A mean outside the declared bounds means one of the three is wrong, and which
	// one is not guessable from here.
	if d.Max > 0 && d.Mean > d.Max {
		p.Field(at+".tokens", "mean %d exceeds tokens_max %d", d.Mean, d.Max)
	}
	if d.Min > 0 && d.Mean > 0 && d.Mean < d.Min {
		p.Field(at+".tokens", "mean %d is below tokens_min %d", d.Mean, d.Min)
	}
}
