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
	"regexp"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// An address in the range LiteLLM's load balancer uses (k8s/agents/network.yaml).
var litellmProbe = net.ParseIP("9.0.0.1")

type labelSelector struct {
	MatchLabels      map[string]string `yaml:"matchLabels"`
	MatchExpressions []struct {
		Key      string   `yaml:"key"`
		Operator string   `yaml:"operator"`
		Values   []string `yaml:"values"`
	} `yaml:"matchExpressions"`
}

// matches reports whether the selector selects a pod (or namespace) with these labels. An empty
// selector selects everything.
func (s labelSelector) matches(labels map[string]string) bool {
	for k, v := range s.MatchLabels {
		if labels[k] != v {
			return false
		}
	}
	for _, e := range s.MatchExpressions {
		v, has := labels[e.Key]
		in := false
		for _, want := range e.Values {
			in = in || (has && v == want)
		}
		switch e.Operator {
		case "In":
			if !in {
				return false
			}
		case "NotIn":
			if in {
				return false
			}
		case "Exists":
			if !has {
				return false
			}
		case "DoesNotExist":
			if has {
				return false
			}
		}
	}
	return true
}

type policyPeer struct {
	PodSelector       *labelSelector `yaml:"podSelector"`
	NamespaceSelector *labelSelector `yaml:"namespaceSelector"`
	IPBlock           *struct {
		CIDR   string   `yaml:"cidr"`
		Except []string `yaml:"except"`
	} `yaml:"ipBlock"`
}

// selectsPod reports whether a peer selects a pod in blis with these labels. A pod selector
// alone means this namespace; a namespace selector alone means every pod in the namespaces it
// selects. An ipBlock does not select pods.
func (p policyPeer) selectsPod(labels map[string]string) bool {
	if p.PodSelector == nil && p.NamespaceSelector == nil {
		return false
	}
	if p.NamespaceSelector != nil && !p.NamespaceSelector.matches(blisNamespaceLabels) {
		return false
	}
	return p.PodSelector == nil || p.PodSelector.matches(labels)
}

type policyPort struct {
	Protocol string `yaml:"protocol"`
	Port     any    `yaml:"port"`
}

// coversPort reports whether a rule's ports include this TCP port (no ports means every port).
func coversPort(ports []policyPort, port int) bool {
	if len(ports) == 0 {
		return true
	}
	for _, p := range ports {
		if (p.Protocol == "" || p.Protocol == "TCP") && (p.Port == nil || p.Port == port) {
			return true
		}
	}
	return false
}

type k8sPolicy struct {
	Metadata struct {
		Name string `yaml:"name"`
	} `yaml:"metadata"`
	Spec struct {
		PodSelector labelSelector `yaml:"podSelector"`
		Ingress     []struct {
			From  []policyPeer `yaml:"from"`
			Ports []policyPort `yaml:"ports"`
		} `yaml:"ingress"`
		Egress []struct {
			To    []policyPeer `yaml:"to"`
			Ports []policyPort `yaml:"ports"`
		} `yaml:"egress"`
	} `yaml:"spec"`
}

// The proxy pod's labels (k8s/agents/litellm-proxy.yaml), and the blis namespace's own.
var (
	proxyPodLabels      = map[string]string{"app": "litellm-proxy"}
	blisNamespaceLabels = map[string]string{"kubernetes.io/metadata.name": "blis"}
)

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

// selects reports whether a policy selects exactly the pods with this one label.
func selects(p k8sPolicy, key, value string) bool {
	sel := p.Spec.PodSelector
	return len(sel.MatchExpressions) == 0 && len(sel.MatchLabels) == 1 && sel.MatchLabels[key] == value
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

// No pod in blis may reach the cloud metadata endpoint, which hands out the node's credentials.
func TestBlisPoliciesCloseTheMetadataEndpoint(t *testing.T) {
	metadata := net.ParseIP("169.254.169.254")
	for _, p := range blisPolicies(t) {
		for _, rule := range p.Spec.Egress {
			for _, to := range rule.To {
				if to.IPBlock != nil && cidrOpens(t, to.IPBlock.CIDR, to.IPBlock.Except, metadata) {
					t.Errorf("NetworkPolicy %q opens %s to pods %v", p.Metadata.Name, metadata, p.Spec.PodSelector.MatchLabels)
				}
			}
		}
	}
}

// The proxy takes connections from pr-review pods alone, and only they are given a way to it. A
// policy or peer with an empty pod selector selects the proxy too (the old namespace's
// allow-same-namespace was one), so selectors are matched against the proxy's labels.
func TestBlisPoliciesLetOnlyPrReviewReachTheProxy(t *testing.T) {
	ingressRules := 0
	for _, p := range blisPolicies(t) {
		if p.Spec.PodSelector.matches(proxyPodLabels) {
			for _, rule := range p.Spec.Ingress {
				ingressRules++
				if len(rule.From) == 0 {
					t.Errorf("NetworkPolicy %q admits every source to the proxy (an ingress rule with no from)", p.Metadata.Name)
				}
				for _, from := range rule.From {
					pod := from.PodSelector
					if from.IPBlock != nil || from.NamespaceSelector != nil || pod == nil ||
						len(pod.MatchExpressions) != 0 || len(pod.MatchLabels) != 1 || pod.MatchLabels["blis-runner"] != "pr-review" {
						t.Errorf("NetworkPolicy %q admits a peer other than blis-runner=pr-review to the proxy", p.Metadata.Name)
					}
				}
			}
		}
		if selects(p, "blis-runner", "pr-review") {
			continue
		}
		for _, rule := range p.Spec.Egress {
			if !coversPort(rule.Ports, 4000) {
				continue
			}
			if len(rule.To) == 0 {
				t.Errorf("NetworkPolicy %q gives pods %v a way to the proxy (an egress rule with no to)",
					p.Metadata.Name, p.Spec.PodSelector.MatchLabels)
			}
			for _, to := range rule.To {
				if to.selectsPod(proxyPodLabels) {
					t.Errorf("NetworkPolicy %q gives pods %v a way to the proxy", p.Metadata.Name, p.Spec.PodSelector.MatchLabels)
				}
			}
		}
	}
	if ingressRules == 0 {
		t.Error("no ingress rule admits the pr-review pods to the proxy")
	}
}

// The proxy relays only the routes the reviewers call, forwards the path it matched, and checks
// the upstream's certificate before it sends the key.
func TestBlisLiteLLMProxyRoutesAndTLS(t *testing.T) {
	var conf string
	dec := yaml.NewDecoder(strings.NewReader(readFileOrFail(t, filepath.Join("..", "k8s", "agents", "litellm-proxy.yaml"))))
	for {
		var doc struct {
			Kind string            `yaml:"kind"`
			Data map[string]string `yaml:"data"`
		}
		err := dec.Decode(&doc)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("decoding k8s/agents/litellm-proxy.yaml: %v", err)
		}
		if doc.Kind == "ConfigMap" {
			conf = doc.Data["default.conf.template"]
		}
	}
	if conf == "" {
		t.Fatal("no default.conf.template in the litellm-proxy ConfigMap")
	}
	relays, catchAll := 0, false
	for _, loc := range nginxLocations(t, conf) {
		if !strings.Contains(loc.body, "proxy_pass") {
			if loc.args == "/" {
				catchAll = strings.Contains(loc.body, "return 404;")
			}
			continue
		}
		relays++
		if !strings.HasPrefix(loc.args, "~ ") {
			t.Fatalf("location %q relays with the key; only the regex allowlist may", loc.args)
		}
		re, err := regexp.Compile(strings.TrimPrefix(loc.args, "~ "))
		if err != nil {
			t.Fatalf("location %q: %v", loc.args, err)
		}
		for path, want := range map[string]bool{
			"/v1/messages": true, "/v1/messages/count_tokens": true, "/v1/models": true,
			"/chat/completions": true, "/v1/chat/completions": true,
			"/": false, "/models": false, "/v1/models/x": false, "/v1/messages/batches": false,
			"/key/generate": false, "/user/info": false, "/spend/logs": false, "/v1/messages/x/count_tokens": false,
		} {
			if got := re.MatchString(path); got != want {
				t.Errorf("location %q: relays %s = %v, want %v", loc.args, path, got, want)
			}
		}
		for _, d := range []string{
			"set $litellm_upstream $LITELLM_UPSTREAM;",
			"proxy_pass $litellm_upstream$uri$is_args$args;",
			"proxy_ssl_server_name on;",
			"proxy_ssl_verify on;",
			"proxy_ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;",
		} {
			if !strings.Contains(loc.body, d) {
				t.Errorf("location %q: missing %q", loc.args, d)
			}
		}
	}
	if relays != 1 {
		t.Errorf("%d locations relay to LiteLLM, want the one allowlist", relays)
	}
	if !catchAll {
		t.Error("no `location /` returning 404 for the routes the allowlist leaves out")
	}
}

type nginxLocation struct{ args, body string }

// nginxLocations returns each `location <args> { <body> }` in an nginx config template, comments
// dropped and each envsubst ${NAME} read as $NAME, with no nested blocks inside a location.
func nginxLocations(t *testing.T, conf string) []nginxLocation {
	t.Helper()
	lines := strings.Split(conf, "\n")
	for i, l := range lines {
		if c := strings.Index(l, "#"); c >= 0 {
			lines[i] = l[:c]
		}
	}
	envsubst := regexp.MustCompile(`\$\{(\w+)\}`)
	var out []nginxLocation
	for rest := envsubst.ReplaceAllString(strings.Join(lines, "\n"), "$$$1"); ; {
		i := strings.Index(rest, "location ")
		if i < 0 {
			return out
		}
		rest = rest[i+len("location "):]
		open, end := strings.Index(rest, "{"), strings.Index(rest, "}")
		if open < 0 || end < open {
			t.Fatalf("unbalanced location block near %q", rest)
		}
		out = append(out, nginxLocation{args: strings.TrimSpace(rest[:open]), body: rest[open+1 : end]})
		rest = rest[end+1:]
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
		// The policies also trust these labels (the proxy's, and allow-arc-api-server's).
		for _, trusted := range []string{"app", "app.kubernetes.io/component"} {
			if got, set := v.Template.Metadata.Labels[trusted]; set {
				t.Errorf("k8s/%s: sets pod label %s=%q, which a NetworkPolicy trusts", f, trusted, got)
			}
		}
		if tok := v.Template.Spec.AutomountServiceAccountToken; tok == nil || *tok {
			t.Errorf("k8s/%s: runner pods must set automountServiceAccountToken: false", f)
		}
	}
}

// Every job that holds the agent credentials runs on the agent runners, and nothing else does:
// the untrusted reviewer has its own pool (pr_review_test.go), and no other job names it.
func TestAgentJobsRunOnTheAgentRunners(t *testing.T) {
	agent := map[string]int{
		"claude.yml": 2, "archon.yml": 1, "deliver-implement.yml": 1, "deliver-verify.yml": 1, "deliver-correct.yml": 1,
	}
	files, err := filepath.Glob(filepath.Join("..", ".github", "workflows", "*.yml"))
	if err != nil {
		t.Fatal(err)
	}
	holds := func(n yaml.Node) bool {
		return strings.Contains(strings.Join(flattenScalars(n), "\n"), "secrets.LITELLM_")
	}
	for _, f := range files {
		var wf map[string]yaml.Node
		body, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if err := yaml.Unmarshal(body, &wf); err != nil {
			t.Fatalf("parsing %s: %v", f, err)
		}
		var jobs map[string]yaml.Node
		if n, ok := wf["jobs"]; ok {
			if err := n.Decode(&jobs); err != nil {
				t.Fatalf("parsing %s jobs: %v", f, err)
			}
		}
		// A workflow-level reference (env:) reaches every job.
		fileHolds := false
		for key, n := range wf {
			fileHolds = fileHolds || (key != "jobs" && holds(n))
		}
		name, got := filepath.Base(f), 0
		for job, n := range jobs {
			var j struct {
				RunsOn yaml.Node `yaml:"runs-on"`
			}
			if err := n.Decode(&j); err != nil {
				t.Fatalf("parsing %s job %q: %v", name, job, err)
			}
			raw := strings.Join(flattenScalars(j.RunsOn), ",")
			if strings.Contains(raw, agentRunsOn) || strings.Contains(raw, "self-hosted") {
				got++
				if raw != agentRunsOn {
					t.Errorf("%s job %q: runs-on %q, want %q", name, job, raw, agentRunsOn)
				}
			}
			if (fileHolds || holds(n)) && raw != agentRunsOn {
				t.Errorf("%s job %q references secrets.LITELLM_* but runs on %q, want %q", name, job, raw, agentRunsOn)
			}
			if strings.Contains(raw, "blis-ci-pr-review") && (name != "pr-review.yml" || job != "review") {
				t.Errorf("%s job %q runs on blis-ci-pr-review, which is pr-review.yml's review job's alone", name, job)
			}
		}
		if got != agent[name] {
			t.Errorf("%s: %d jobs on the agent runners, want %d", name, got, agent[name])
		}
	}
}
