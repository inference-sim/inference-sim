package cmd

import (
	"runtime/debug"
	"testing"
)

func TestDepVersionPin(t *testing.T) {
	defer func(old string) { buildDepPins = old }(buildDepPins)
	buildDepPins = "example.com/a=v1.2.3;example.com/b=v9.9.9"
	local := &debug.Module{Path: "example.com/a", Version: "v0.0.0-0", Replace: &debug.Module{Path: "./x", Version: "(devel)"}}
	if got := depVersion(local); got != "v1.2.3" {
		t.Fatalf("a locally replaced module with a pin reports %q, want its pin", got)
	}
	unpinned := &debug.Module{Path: "example.com/c", Version: "v0.0.0-0", Replace: &debug.Module{Path: "./x", Version: "(devel)"}}
	if got := depVersion(unpinned); got != "(devel)" {
		t.Fatalf("an unpinned local replace reports %q, want the unchanged answer", got)
	}
	versioned := &debug.Module{Path: "example.com/a", Version: "v0.0.0-0", Replace: &debug.Module{Path: "example.com/fork", Version: "v2.0.0"}}
	if got := depVersion(versioned); got != "v2.0.0" {
		t.Fatalf("a replacement by a versioned module reports %q, want v2.0.0", got)
	}
}
