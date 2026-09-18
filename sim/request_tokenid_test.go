package sim_test

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestTokenID_DefinedType_NotAlias verifies BC-1: TokenID is a defined type, not
// a transparent alias for int32. A raw int32 must NOT be assignable to a TokenID
// without an explicit conversion — the compile-time mixed-arithmetic check is the
// whole point of this PR.
func TestTokenID_DefinedType_NotAlias(t *testing.T) {
	var x sim.TokenID
	rt := reflect.TypeOf(x)
	if rt.Kind() != reflect.Int32 {
		t.Fatalf("TokenID kind = %v, want int32", rt.Kind())
	}
	if rt.Name() != "TokenID" {
		t.Fatalf("TokenID type name = %q, want %q", rt.Name(), "TokenID")
	}
	// Defined-type identity: the reflect.Type for TokenID must differ from int32.
	// (A Go type alias `type TokenID = int32` would make these reflect.Types equal.)
	if rt == reflect.TypeOf(int32(0)) {
		t.Fatalf("TokenID must be a defined type, not a transparent alias of int32")
	}
}

// TestRequest_TokenFields_AreTokenIDSlices verifies BC-2: Request.InputTokens
// and Request.OutputTokens are typed []TokenID.
func TestRequest_TokenFields_AreTokenIDSlices(t *testing.T) {
	var req sim.Request
	rt := reflect.TypeOf(req)
	for _, name := range []string{"InputTokens", "OutputTokens"} {
		f, ok := rt.FieldByName(name)
		if !ok {
			t.Fatalf("Request.%s field not found", name)
		}
		if f.Type.Kind() != reflect.Slice {
			t.Fatalf("Request.%s kind = %v, want slice", name, f.Type.Kind())
		}
		elem := f.Type.Elem()
		if elem.Name() != "TokenID" {
			t.Errorf("Request.%s element type = %q, want %q", name, elem.Name(), "TokenID")
		}
	}
}

// TestGenerateRandomTokenIDs_ReturnsTokenIDSlice verifies BC-3: the generator
// returns []TokenID and respects the [0, MaxTokenID) range.
func TestGenerateRandomTokenIDs_ReturnsTokenIDSlice(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	got := sim.GenerateRandomTokenIDs(rng, 8)
	if len(got) != 8 {
		t.Fatalf("len = %d, want 8", len(got))
	}
	if reflect.TypeOf(got).Elem().Name() != "TokenID" {
		t.Fatalf("element type = %q, want TokenID", reflect.TypeOf(got).Elem().Name())
	}
	for i, v := range got {
		if int(v) < 0 || int(v) >= sim.MaxTokenID {
			t.Errorf("got[%d] = %d, out of [0, %d)", i, v, sim.MaxTokenID)
		}
	}
}

// TestTokenID_HoldsValuesNearInt32Max guards against accidental narrowing of
// the underlying type (e.g., to int16). MaxTokenID is well below MaxInt32, but
// TokenID must be representable up to MaxInt32 to satisfy the issue's invariant
// "vocabulary sizes never approach 2^31".
func TestTokenID_HoldsValuesNearInt32Max(t *testing.T) {
	var maxTok sim.TokenID = math.MaxInt32
	if int64(maxTok) != int64(math.MaxInt32) {
		t.Fatalf("TokenID cannot hold MaxInt32: got %d", maxTok)
	}
}
