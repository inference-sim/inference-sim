// Package tokenid defines the simulator's compact token-ID type.
// Kept in a tiny leaf package so any sim/* package can import it without
// risking an import cycle through sim itself.
package tokenid

// TokenID is the simulator's representation of a tokenizer vocabulary ID.
// 32 bits is sufficient for all real LLM vocabularies (the simulator's
// MaxTokenID is 128000, well below 2^31). It is a defined type, NOT a
// type alias, so mixing with int requires an explicit conversion — the
// compile-time check is the point.
type TokenID int32
