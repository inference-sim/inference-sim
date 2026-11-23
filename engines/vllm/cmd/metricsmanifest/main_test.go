package main

import (
	"strings"
	"testing"
)

// A miniature release: one logger family, one name-constant family bound by a metadata
// table, one spec-decode table entry, and one literal that is only a lookup.
var tree = map[string]string{
	"vllm/v1/metrics/loggers.py": `class PrometheusStatLogger:
    def __init__(self):
        gauge = self._gauge_cls(
            name="vllm:num_requests_running",
            documentation="Running.",
            labelnames=labelnames,
        )
`,
	"vllm/v1/kv_offload/cpu/common.py": `class CPUOffloadingMetrics:
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
`,
	"vllm/v1/kv_offload/cpu/spec.py": `def defs():
    return {
        CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
            documentation="usage",
        ),
    }
`,
	"vllm/v1/spec_decode/metrics.py": `counter_specs = [
    ("vllm:spec_decode_num_drafts", "Number of drafts."),
]
`,
	"vllm/v1/metrics/reader.py": `if metric.name == "vllm:num_requests_running":
    pass
`,
}

func classified() classesFile {
	c := classesFile{VLLM: "test", NotFamilies: map[string]string{
		"vllm/v1/metrics/reader.py:vllm:num_requests_running": "a lookup",
	}, Families: map[string]classification{}}
	for _, n := range []string{"vllm:num_requests_running", "vllm:kv_offload_cpu_cache_usage_perc", "vllm:spec_decode_num_drafts"} {
		c.Families[n] = classification{Class: "exact", Reason: "r"}
	}
	return c
}

func TestEveryDefinitionFormIsFound(t *testing.T) {
	m, problems := Build(tree, classified())
	if len(problems) > 0 {
		t.Fatalf("problems: %v", problems)
	}
	want := map[string]string{
		"vllm:num_requests_running":            "gauge",
		"vllm:kv_offload_cpu_cache_usage_perc": "gauge",
		"vllm:spec_decode_num_drafts":          "counter",
	}
	if len(m.Families) != len(want) {
		t.Fatalf("%d families, want %d: %+v", len(m.Families), len(want), m.Families)
	}
	for _, f := range m.Families {
		if want[f.Name] != f.Type {
			t.Errorf("%s: type %q, want %q", f.Name, f.Type, want[f.Name])
		}
	}
	if m.Counts.Total != 3 || m.Counts.Exact != 3 {
		t.Errorf("counts %+v", m.Counts)
	}
}

// Whatever a release adds, drops or leaves unexplained fails the run, naming it.
func TestAReleaseChangeFailsUntilClassified(t *testing.T) {
	for _, c := range []struct {
		name string
		edit func(map[string]string, *classesFile)
		want string
	}{
		{"a new family", func(f map[string]string, _ *classesFile) {
			f["vllm/v1/metrics/new.py"] = "c = Counter(\n    name=\"vllm:new_family\",\n)\n"
		}, "vllm:new_family (counter, vllm/v1/metrics/new.py:2) has no classification"},
		{"a dropped family", func(f map[string]string, _ *classesFile) {
			delete(f, "vllm/v1/spec_decode/metrics.py")
		}, "classifies vllm:spec_decode_num_drafts, which the release does not define"},
		{"an unexplained literal", func(f map[string]string, _ *classesFile) {
			f["vllm/benchmarks/x.py"] = "if line.startswith(\"vllm:foo\"):\n    pass\n"
		}, "vllm/benchmarks/x.py:1: vllm:foo defines no family"},
		{"a stale not_families entry", func(f map[string]string, _ *classesFile) {
			delete(f, "vllm/v1/metrics/reader.py")
		}, "not_families lists vllm/v1/metrics/reader.py:vllm:num_requests_running"},
		{"a name constant never typed", func(f map[string]string, _ *classesFile) {
			delete(f, "vllm/v1/kv_offload/cpu/spec.py")
		}, "CPU_CACHE_USAGE_PERC = vllm:kv_offload_cpu_cache_usage_perc is never bound to a metric type"},
		{"a classification with no reason", func(_ map[string]string, c *classesFile) {
			c.Families["vllm:spec_decode_num_drafts"] = classification{Class: "exact"}
		}, "classified exact with no reason"},
		{"an unknown class", func(_ map[string]string, c *classesFile) {
			c.Families["vllm:spec_decode_num_drafts"] = classification{Class: "maybe", Reason: "r"}
		}, `class "maybe" is not exact, derived or unavailable`},
	} {
		t.Run(c.name, func(t *testing.T) {
			files := map[string]string{}
			for k, v := range tree {
				files[k] = v
			}
			classes := classified()
			c.edit(files, &classes)
			_, problems := Build(files, classes)
			for _, p := range problems {
				if strings.Contains(p, c.want) {
					return
				}
			}
			t.Errorf("no problem naming %q; got %v", c.want, problems)
		})
	}
}
