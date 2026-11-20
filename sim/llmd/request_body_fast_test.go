package llmd

import (
	"bytes"
	"encoding/json"
	"testing"
)

// The fast request body is the bytes json.Marshal writes for the same map.
func TestTextRequestBodyEqualsMarshal(t *testing.T) {
	for _, model := range []string{"m", "deepseek-v4.1-flash", `we"ird<&> `} {
		for _, prompt := range [][]uint32{{}, {0}, {1, 2, 4294967295}, make([]uint32, 300)} {
			for _, stream := range []bool{false, true} {
				for _, mt := range []int{0, 1, 512} {
					want := map[string]any{"model": model, "prompt": prompt, "stream": stream}
					if mt > 0 {
						want["max_tokens"] = mt
					}
					w, _ := json.Marshal(want)
					if g := textRequestBody(model, prompt, stream, mt); !bytes.Equal(g, w) {
						t.Fatalf("got %s want %s", g, w)
					}
				}
			}
		}
	}
}
