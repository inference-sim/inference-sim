package router

import (
	"strings"
	"testing"
)

func epp() *Router {
	return &Router{Kind: KindEPP, EPP: &EPP{Version: "v0.2.1",
		Config: ConfigRef{Path: "plugins-v2.yaml", SHA256: strings.Repeat("a", 64)}}}
}

func TestValidRoutersPass(t *testing.T) {
	for name, r := range map[string]*Router{"epp": epp(), "direct": {Kind: KindDirect}} {
		if p := r.Validate(); !p.OK() {
			t.Errorf("%s: a valid router was rejected:\n%s", name, p.Error())
		}
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name, path string
		mutate     func(*Router)
	}{
		{"no kind", "kind", func(r *Router) { r.Kind = "" }},
		{"unknown kind", "kind", func(r *Router) { r.Kind = "haproxy" }},
		{"epp without a block", "epp", func(r *Router) { r.EPP = nil }},
		{"direct with a picker", "epp", func(r *Router) { r.Kind = KindDirect }},
		{"no version", "epp.version", func(r *Router) { r.EPP.Version = "" }},
		{"no config path", "epp.config.path", func(r *Router) { r.EPP.Config.Path = "" }},
		{"no config digest", "epp.config.sha256", func(r *Router) { r.EPP.Config.SHA256 = "" }},
		{"short config digest", "epp.config.sha256", func(r *Router) { r.EPP.Config.SHA256 = "abc" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := epp()
			tc.mutate(r)
			for _, pr := range r.Validate().Errors() {
				if pr.Path == tc.path {
					return
				}
			}
			t.Fatalf("expected a problem at %s:\n%s", tc.path, r.Validate().Error())
		})
	}
}
