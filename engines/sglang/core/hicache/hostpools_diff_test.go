// Original to blis-sglang.

package hicache

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// HostPoolsGoldenDir holds the traces oracle/run_host_pools.py records at SGLang's own MHA, MLA and
// DSA host pools, built as HiCache's assembler builds them.
const HostPoolsGoldenDir = "testdata/golden-fid5/host-pools"

type hostScenario struct {
	Pools []struct {
		As    string         `json:"as"`
		Class string         `json:"class"`
		Args  map[string]any `json:"args"`
	} `json:"pools"`
	Hosts []struct {
		As       string  `json:"as"`
		Pool     string  `json:"pool"`
		Layout   string  `json:"layout"`
		Ratio    float64 `json:"ratio"`
		PageSize int     `json:"page_size"`
		Raises   string  `json:"raises"`
	} `json:"hosts"`
	Indexers []struct {
		As     string `json:"as"`
		Pool   string `json:"pool"`
		Anchor string `json:"anchor"`
		Raises string `json:"raises"`
	} `json:"indexers"`
}

type hostTrace struct {
	Hosts []struct {
		As           string `json:"as"`
		Class        string `json:"class"`
		Layout       string `json:"layout"`
		Size         int    `json:"size"`
		PageNum      int    `json:"page_num"`
		PageSize     int    `json:"page_size"`
		SizePerToken int64  `json:"size_per_token"`
		LayerNum     int    `json:"layer_num"`
		Raises       string `json:"raises"`
		Message      string `json:"message"`
	} `json:"hosts"`
}

func num(a map[string]any, k string, def int) int {
	if v, ok := a[k].(float64); ok {
		return int(v)
	}
	return def
}

func itemOf(a map[string]any) int {
	switch a["dtype"].(map[string]any)["dtype"] {
	case "float8_e4m3fn", "float8_e5m2":
		return 1
	case "float32":
		return 4
	}
	return 2
}

// Every host pool SGLang's assembler built for a device pool, built again by the port from the
// device pool's arguments, takes the bytes per token, the slots and the pages SGLang's takes, and
// a layout the pool cannot lay out is refused where SGLang refuses it, with its message.
func TestHostPoolSizingIsSGLangs(t *testing.T) {
	paths, _ := filepath.Glob(filepath.Join(HostPoolsGoldenDir, "*.trace.json"))
	if len(paths) == 0 {
		t.Fatalf("no trace under %s", HostPoolsGoldenDir)
	}
	checked, refused := 0, 0
	for _, p := range paths {
		name := strings.TrimSuffix(filepath.Base(p), ".trace.json")
		var sc hostScenario
		var tr hostTrace
		for path, into := range map[string]any{filepath.Join("..", "..", "oracle", "scenarios-host-pools", name+".json"): &sc, p: &tr} {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, into); err != nil {
				t.Fatalf("%s: %v", path, err)
			}
		}
		pools := map[string]map[string]any{}
		class := map[string]string{}
		for _, d := range sc.Pools {
			pools[d.As], class[d.As] = d.Args, d.Class
		}
		shapes := map[string]HostShape{}
		i := 0
		for _, h := range sc.Hosts {
			want := tr.Hosts[i]
			i++
			a := pools[h.Pool]
			var s HostShape
			switch class[h.Pool] {
			case "MHATokenToKVPool":
				s = MHAHost(num(a, "head_num", 0), num(a, "head_dim", 0), num(a, "v_head_dim", num(a, "head_dim", 0)), num(a, "layer_num", 0), itemOf(a), 1)
			case "MLATokenToKVPool":
				s = MLAHost(num(a, "kv_lora_rank", 0), num(a, "qk_rope_head_dim", 0), 0, num(a, "layer_num", 0), itemOf(a))
			case "DSATokenToKVPool":
				s = MLAHost(num(a, "kv_lora_rank", 0), num(a, "qk_rope_head_dim", 0), num(a, "kv_cache_dim", 0), num(a, "layer_num", 0), itemOf(a))
			}
			if err := s.CheckLayout(h.Layout); err != nil {
				if want.Raises == "" || !strings.Contains(want.Message, err.Error()) && !strings.Contains(err.Error(), want.Message) {
					t.Errorf("%s/%s: the port refuses %q (%v); SGLang %q %q", name, h.As, h.Layout, err, want.Raises, want.Message)
				}
				refused++
				continue
			}
			if want.Raises != "" {
				t.Errorf("%s/%s: SGLang refuses %s (%s: %s), the port builds it", name, h.As, h.Layout, want.Raises, want.Message)
				continue
			}
			shapes[h.As] = s
			hp, err := NewHostPool(HostSizing{DeviceCapacityTokens: num(a, "size", 0), SizePerToken: s.SizePerToken(), HostToDeviceRatio: h.Ratio,
				PageSize: h.PageSize, Layout: h.Layout})
			if err != nil {
				t.Fatal(err)
			}
			if s.SizePerToken() != want.SizePerToken || hp.Size() != want.Size || hp.PageNum() != want.PageNum || s.LayerNum != want.LayerNum {
				t.Errorf("%s/%s: %d bytes per token, %d slots, %d pages, %d layers; SGLang %d, %d, %d, %d", name, h.As, s.SizePerToken(),
					hp.Size(), hp.PageNum(), s.LayerNum, want.SizePerToken, want.Size, want.PageNum, want.LayerNum)
			}
			checked++
		}
		for _, x := range sc.Indexers {
			want := tr.Hosts[i]
			i++
			a := pools[x.Pool]
			owned := num(a, "layer_num", 0)
			if skip, ok := a["skip_topk_layers"].([]any); ok {
				owned = 0
				for _, v := range skip {
					if b, _ := v.(bool); !b {
						owned++
					}
				}
			}
			anchorPage := 0
			for _, h := range sc.Hosts {
				if h.As == x.Anchor {
					anchorPage = h.PageSize
				}
			}
			s, err := DSAIndexerHost(num(a, "index_head_dim", 0), 128, anchorPage, num(a, "index_kpool", 1), owned)
			if err != nil {
				if want.Raises == "" {
					t.Errorf("%s/%s: the port refuses (%v), SGLang builds it", name, x.As, err)
				}
				refused++
				continue
			}
			if s.SizePerToken() != want.SizePerToken || s.LayerNum != want.LayerNum {
				t.Errorf("%s/%s: %d bytes per token over %d layers; SGLang %d over %d", name, x.As, s.SizePerToken(), s.LayerNum,
					want.SizePerToken, want.LayerNum)
			}
			checked++
		}
	}
	if checked == 0 || refused == 0 {
		t.Fatalf("checked %d built pools and %d refusals", checked, refused)
	}
}
