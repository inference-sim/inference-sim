package workload

import "testing"

// A ceiling on a round's prompt plus output (inference-sim#1766): wekaThreeTurnWithSubagent has
// rounds of 110, 170 and 220 tokens. A session with a round over the ceiling is dropped whole
// (drop-session) or kept up to the first such round (truncate); the rounds kept are untouched,
// so their prefix chain is intact.
func TestConvertWekaSession_MaxContext(t *testing.T) {
	convert := func(max int, mode MaxContextMode) int {
		recs, err := ConvertWekaSession([]byte(wekaThreeTurnWithSubagent),
			WekaConvertOptions{MinRounds: 1, DropSubagents: true, MaxContext: max, MaxContextMode: mode})
		if err != nil {
			t.Fatal(err)
		}
		return len(recs)
	}
	for _, c := range []struct {
		max  int
		mode MaxContextMode
		want int
	}{
		{0, "", 3},
		{220, MaxContextDropSession, 3},
		{200, MaxContextDropSession, 0},
		{200, MaxContextTruncate, 2},
		{170, MaxContextTruncate, 2},
		{169, MaxContextTruncate, 1},
		{100, MaxContextTruncate, 0},
	} {
		if got := convert(c.max, c.mode); got != c.want {
			t.Errorf("ceiling %d, %q: %d rounds, want %d", c.max, c.mode, got, c.want)
		}
	}
	// A truncated session below --min-rounds is dropped like any short session.
	recs, err := ConvertWekaSession([]byte(wekaThreeTurnWithSubagent),
		WekaConvertOptions{MinRounds: 2, DropSubagents: true, MaxContext: 169, MaxContextMode: MaxContextTruncate})
	if err != nil || recs != nil {
		t.Errorf("a one-round truncation under --min-rounds 2 gave %d records, err %v; want none", len(recs), err)
	}
}

// A ceiling needs a stated mode, and a mode needs a ceiling.
func TestValidateMaxContext(t *testing.T) {
	for _, c := range []struct {
		max  int
		mode MaxContextMode
		ok   bool
	}{
		{0, "", true}, {4096, MaxContextDropSession, true}, {4096, MaxContextTruncate, true},
		{4096, "", false}, {4096, "bogus", false}, {0, MaxContextTruncate, false}, {-1, "", false},
	} {
		if err := ValidateMaxContext(c.max, c.mode); (err == nil) != c.ok {
			t.Errorf("ValidateMaxContext(%d, %q) = %v, want ok=%v", c.max, c.mode, err, c.ok)
		}
	}
}
