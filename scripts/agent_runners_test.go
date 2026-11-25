package scripts_test

// Guard tests for the agent runner scale sets (k8s/agents). They share the blis namespace with the
// CI scale sets, which run pull request code, so what keeps the two apart is a pod label: only
// the agent pods may reach LiteLLM, and only the pr-review pods may reach the litellm-proxy.
// These parse the manifests and the workflows and check those laws, not the text.

import (
	"errors"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// An address in the range LiteLLM's load balancer uses (k8s/agents/network.yaml).
var litellmProbe = net.ParseIP("9.0.0.1")

type k8sPolicy struct {
	Metadata struct {
		Name string `yaml:"name"`
	} `yaml:"metadata"`
	Spec struct {
		PodSelector struct {
			MatchLabels map[string]string `yaml:"matchLabels"`
		} `yaml:"podSelector"`
		Ingress []struct {
			From []struct {
				PodSelector *struct {
					MatchLabels map[string]string `yaml:"matchLabels"`
				} `yaml:"podSelector"`
				IPBlock *struct {
					CIDR string `yaml:"cidr"`
				} `yaml:"ipBlock"`
			} `yaml:"from"`
		} `yaml:"ingress"`
		Egress []struct {
			To []struct {
				PodSelector *struct {
					MatchLabels map[string]string `yaml:"matchLabels"`
				} `yaml:"podSelector"`
				IPBlock *struct {
					CIDR   string   `yaml:"cidr"`
					Except []string `yaml:"except"`
				} `yaml:"ipBlock"`
			} `yaml:"to"`
		} `yaml:"egress"`
	} `yaml:"spec"`
}

func blisPolicies(t *testing.T) []k8sPolicy {
	t.Helper()
	var out []k8sPolicy
	for _, f := range []string{"ci/network.yaml", "agents/network.yaml"} {
		dec := yaml.NewDecoder(strings.NewReader(readFileOrFail(t, filepath.Join("..", "k8s", f))))
		for {
			var doc yaml.Node
			err := dec.Decode(&doc)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatalf("decoding k8s/%s: %v", f, err)
			}
			var kind struct {
				Kind string `yaml:"kind"`
			}
			if err := doc.Decode(&kind); err != nil || kind.Kind != "NetworkPolicy" {
				continue
			}
			var p k8sPolicy
			if err := doc.Decode(&p); err != nil {
				t.Fatalf("decoding a NetworkPolicy in k8s/%s: %v", f, err)
			}
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		t.Fatal("no NetworkPolicy found in k8s/ci or k8s/agents")
	}
	return out
}

func cidrOpens(t *testing.T, cidr string, except []string, ip net.IP) bool {
	t.Helper()
	_, n, err := net.ParseCIDR(cidr)
	if err != nil {
		t.Fatalf("bad cidr %q: %v", cidr, err)
	}
	if !n.Contains(ip) {
		return false
	}
	for _, e := range except {
		if _, en, err := net.ParseCIDR(e); err == nil && en.Contains(ip) {
			return false
		}
	}
	return true
}

func selects(p k8sPolicy, key, value string) bool {
	return len(p.Spec.PodSelector.MatchLabels) == 1 && p.Spec.PodSelector.MatchLabels[key] == value
}

// The intranet, where LiteLLM is, opens to the agent pods and the proxy alone: never to a policy
// that selects every pod in blis, a CI pod or a pr-review pod.
func TestBlisPoliciesOpenLiteLLMToTheAgentsAndProxyOnly(t *testing.T) {
	opened := map[string]bool{}
	for _, p := range blisPolicies(t) {
		for _, rule := range p.Spec.Egress {
			for _, to := range rule.To {
				if to.IPBlock == nil || !cidrOpens(t, to.IPBlock.CIDR, to.IPBlock.Except, litellmProbe) {
					continue
				}
				switch {
				case selects(p, "blis-runner", "agents"):
					opened["agents"] = true
				case selects(p, "app", "litellm-proxy"):
					opened["proxy"] = true
				default:
					t.Errorf("NetworkPolicy %q opens %s to pods %v; only blis-runner=agents and app=litellm-proxy may reach LiteLLM",
						p.Metadata.Name, litellmProbe, p.Spec.PodSelector.MatchLabels)
				}
			}
		}
	}
	for _, who := range []string{"agents", "proxy"} {
		if !opened[who] {
			t.Errorf("no NetworkPolicy lets the %s reach LiteLLM's range, which they need", who)
		}
	}
}

// The proxy takes connections from pr-review pods alone, and only they are given a way to it.
func TestBlisPoliciesLetOnlyPrReviewReachTheProxy(t *testing.T) {
	ingressRules := 0
	for _, p := range blisPolicies(t) {
		if selects(p, "app", "litellm-proxy") {
			for _, rule := range p.Spec.Ingress {
				ingressRules++
				for _, from := range rule.From {
					if from.IPBlock != nil || from.PodSelector == nil ||
						len(from.PodSelector.MatchLabels) != 1 || from.PodSelector.MatchLabels["blis-runner"] != "pr-review" {
						t.Errorf("NetworkPolicy %q admits a peer other than blis-runner=pr-review to the proxy", p.Metadata.Name)
					}
				}
			}
		}
		for _, rule := range p.Spec.Egress {
			for _, to := range rule.To {
				if to.PodSelector != nil && to.PodSelector.MatchLabels["app"] == "litellm-proxy" &&
					!selects(p, "blis-runner", "pr-review") {
					t.Errorf("NetworkPolicy %q gives pods %v a way to the proxy", p.Metadata.Name, p.Spec.PodSelector.MatchLabels)
				}
			}
		}
	}
	if ingressRules == 0 {
		t.Error("no ingress rule admits the pr-review pods to the proxy")
	}
}

type scaleSetValues struct {
	RunnerScaleSetName string `yaml:"runnerScaleSetName"`
	Template           struct {
		Metadata struct {
			Labels map[string]string `yaml:"labels"`
		} `yaml:"metadata"`
		Spec struct {
			AutomountServiceAccountToken *bool `yaml:"automountServiceAccountToken"`
		} `yaml:"spec"`
	} `yaml:"template"`
}

// The label the policies trust is set by the agent scale sets alone, and no runner pod gets a
// service account token (a pod that could read the API could read the LiteLLM secret).
func TestScaleSetLabelsAndTokens(t *testing.T) {
	want := map[string]string{
		"agents/values-agents.yaml":    "agents",
		"agents/values-pr-review.yaml": "pr-review",
	}
	files, err := filepath.Glob(filepath.Join("..", "k8s", "ci", "values-*.yaml"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no k8s/ci/values-*.yaml found (%v)", err)
	}
	for _, f := range files {
		want["ci/"+filepath.Base(f)] = ""
	}
	for f, label := range want {
		var v scaleSetValues
		body, err := os.ReadFile(filepath.Join("..", "k8s", f))
		if err != nil {
			t.Fatalf("reading k8s/%s: %v", f, err)
		}
		if err := yaml.Unmarshal(body, &v); err != nil {
			t.Fatalf("parsing k8s/%s: %v", f, err)
		}
		if got := v.Template.Metadata.Labels["blis-runner"]; got != label {
			t.Errorf("k8s/%s: pod label blis-runner is %q, want %q", f, got, label)
		}
		if tok := v.Template.Spec.AutomountServiceAccountToken; tok == nil || *tok {
			t.Errorf("k8s/%s: runner pods must set automountServiceAccountToken: false", f)
		}
	}
}

// Every job that holds the agent credentials runs on the agent runners, and nothing else does:
// the untrusted reviewer has its own pool (pr_review_test.go).
func TestAgentJobsRunOnTheAgentRunners(t *testing.T) {
	agent := map[string]int{
		"claude.yml": 2, "archon.yml": 1, "deliver-implement.yml": 1, "deliver-verify.yml": 1, "deliver-correct.yml": 1,
	}
	files, err := filepath.Glob(filepath.Join("..", ".github", "workflows", "*.yml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		var wf struct {
			Jobs map[string]struct {
				RunsOn yaml.Node `yaml:"runs-on"`
			} `yaml:"jobs"`
		}
		body, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if err := yaml.Unmarshal(body, &wf); err != nil {
			t.Fatalf("parsing %s: %v", f, err)
		}
		name, got := filepath.Base(f), 0
		for job, j := range wf.Jobs {
			raw := strings.Join(flattenScalars(j.RunsOn), ",")
			if strings.Contains(raw, "self-hosted") || strings.Contains(raw, "AGENT_RUNS_ON") {
				got++
				if raw != agentRunsOn {
					t.Errorf("%s job %q: runs-on %q, want %q", name, job, raw, agentRunsOn)
				}
			}
		}
		if got != agent[name] {
			t.Errorf("%s: %d jobs on the agent runners, want %d", name, got, agent[name])
		}
	}
}
