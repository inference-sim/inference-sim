// Original to blis-vllm.

package tiering

import (
	"encoding/json"
	"strings"
	"testing"
)

// A request's kv_transfer_params reach the tier as the engine decoded them: with UseNumber, so a
// numeric port or id is a json.Number. The tier reads the same source whichever way a client wrote
// the numbers (Python's json.loads gives an int or a float, and its truthiness reads 0 as unset).
func TestParseSourceReadsNumbersAsTheEngineDecodesThem(t *testing.T) {
	decode := func(raw string) map[string]any {
		dec := json.NewDecoder(strings.NewReader(raw))
		dec.UseNumber()
		var m map[string]any
		if err := dec.Decode(&m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	for _, c := range []struct {
		name, raw string
		want      *srcInfo
	}{
		{"a numeric port and id", `{"remote_prefiller":{"remote_host":"localhost","remote_port":5710,"kv_request_id":7}}`,
			&srcInfo{kvID: "7", peer: "localhost:5710"}},
		{"string port and id", `{"remote_prefiller":{"remote_host":"h","remote_port":"5710","kv_request_id":"kv1"}}`,
			&srcInfo{kvID: "kv1", peer: "h:5710"}},
		{"a zero port is unset", `{"remote_prefiller":{"remote_host":"h","remote_port":0,"kv_request_id":"kv1"}}`, nil},
		{"a zero float id is unset", `{"remote_kv_source":{"remote_host":"h","remote_port":1,"kv_request_id":0.0}}`, nil},
	} {
		got := parseSource(decode(c.raw))
		if (got == nil) != (c.want == nil) || (got != nil && *got != *c.want) {
			t.Errorf("%s: parseSource = %+v, want %+v", c.name, got, c.want)
		}
	}
}
