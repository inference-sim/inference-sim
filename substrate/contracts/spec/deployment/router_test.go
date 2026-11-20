package deployment

import (
	"strings"
	"testing"
)

func TestRouterInputs(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(d *Deployment)
		wantErr string // substring of the problem; "" = valid
	}{
		{"no router block", func(*Deployment) {}, ""},
		{"full router block", func(d *Deployment) {
			d.Router = &Router{
				EPPFlags:             []string{"--pool-name=llama", "--refresh-metrics-interval=50ms"},
				EndpointPickerConfig: "epp-config.yaml",
				Manifests:            []string{"inferencepool.yaml", "objectives.yaml"},
			}
			d.Pools[0].Labels = map[string]string{"app": "llama", "llm-d.ai/role": "prefill"}
		}, ""},
		{"positional argument", func(d *Deployment) { d.Router = &Router{EPPFlags: []string{"llama"}} }, "router.epp_flags[0]"},
		{"configuration named twice", func(d *Deployment) {
			d.Router = &Router{EPPFlags: []string{"--config-file=a.yaml"}, EndpointPickerConfig: "b.yaml"}
		}, "router.endpoint_picker_config"},
		{"configuration named twice, bare flag", func(d *Deployment) {
			d.Router = &Router{EPPFlags: []string{"--config-text", "x"}, EndpointPickerConfig: "b.yaml"}
		}, "router.endpoint_picker_config"},
		{"a flag that only starts like --config-file", func(d *Deployment) {
			d.Router = &Router{EPPFlags: []string{"--config-file-format=yaml"}, EndpointPickerConfig: "b.yaml"}
		}, ""},
		{"empty manifest path", func(d *Deployment) { d.Router = &Router{Manifests: []string{" "}} }, "router.manifests[0]"},
		{"whitespace configuration path", func(d *Deployment) { d.Router = &Router{EndpointPickerConfig: " \t"} }, "router.endpoint_picker_config"},
		{"role label contradicts the role", func(d *Deployment) {
			d.Pools[0].Labels = map[string]string{"llm-d.ai/role": "decode"} // pools[0] is the prefill pool
		}, "pools[0].labels.llm-d.ai/role"},
		{"role label of a colocated pool", func(d *Deployment) {
			d.Pools = d.Pools[:1]
			d.Pools[0].Role = RoleColocated
			d.Pools[0].Labels = map[string]string{"llm-d.ai/role": "prefill"}
		}, "pools[0].labels.llm-d.ai/role"},
		{"bad label key", func(d *Deployment) { d.Pools[1].Labels = map[string]string{"-bad": "x"} }, "pools[1].labels"},
		{"bad label value", func(d *Deployment) { d.Pools[1].Labels = map[string]string{"app": "has space"} }, "pools[1].labels.app"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := pdDeployment()
			tc.mutate(d)
			p := d.Validate()
			problems := ""
			if !p.OK() {
				problems = p.Error()
			}
			if tc.wantErr == "" {
				if problems != "" {
					t.Fatalf("valid deployment rejected: %s", problems)
				}
				return
			}
			if !strings.Contains(problems, tc.wantErr) {
				t.Fatalf("want a problem at %q, got %q", tc.wantErr, problems)
			}
		})
	}
}

func TestRoleLabelValue(t *testing.T) {
	for r, want := range map[Role]string{RoleColocated: "prefill-decode", RolePrefill: "prefill", RoleDecode: "decode"} {
		if got := r.RoleLabelValue(); got != want {
			t.Errorf("%s: llm-d.ai/role %q, want %q", r, got, want)
		}
	}
}

func TestLabelSyntax(t *testing.T) {
	for k, want := range map[string]bool{
		"app": true, "llm-d.ai/role": true, "example.com/a_b.c": true,
		"": false, "/x": false, "-x": false, "x-": false, strings.Repeat("a", 64): false,
		// The prefix is a lowercase DNS-1123 subdomain.
		"Example.com/name": false, ".example.com/name": false, "example..com/name": false,
		"example-.com/name": false, "-example.com/name": false, "ex_ample.com/name": false,
		strings.Repeat("a", 64) + ".com/name": false, strings.Repeat("a", 63) + ".com/name": true,
		strings.Repeat("a.", 127) + "a/name": false, "a-b.c-d.example/name": true,
	} {
		if got := validLabelKey(k); got != want {
			t.Errorf("validLabelKey(%q) = %v, want %v", k, got, want)
		}
	}
	for v, want := range map[string]bool{"": true, "decode": true, "prefill-decode": true, "a b": false, "x.": false} {
		if got := validLabelValue(v); got != want {
			t.Errorf("validLabelValue(%q) = %v, want %v", v, got, want)
		}
	}
}
