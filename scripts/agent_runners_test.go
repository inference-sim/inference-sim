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
	"slices"
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
	EndPort  int    `yaml:"endPort"`
}

// coversPort reports whether a rule's ports include this TCP port: no ports or no port means
// every port, port..endPort is a range, and a named (non-numeric) port counts as covering it,
// since the name may resolve to it.
func coversPort(ports []policyPort, port int) bool {
	if len(ports) == 0 {
		return true
	}
	for _, p := range ports {
		if p.Protocol != "" && p.Protocol != "TCP" {
			continue
		}
		v, numeric := p.Port.(int)
		if !numeric || v == port || (v <= port && port <= p.EndPort) {
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

// egressOpens reports whether an egress rule's peers reach this address: an ipBlock that holds it,
// or no `to` at all, which means every destination.
func egressOpens(t *testing.T, to []policyPeer, ip net.IP) bool {
	t.Helper()
	if len(to) == 0 {
		return true
	}
	for _, peer := range to {
		if peer.IPBlock != nil && cidrOpens(t, peer.IPBlock.CIDR, peer.IPBlock.Except, ip) {
			return true
		}
	}
	return false
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
			if !egressOpens(t, rule.To, litellmProbe) {
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
			if egressOpens(t, rule.To, metadata) {
				t.Errorf("NetworkPolicy %q opens %s to pods %v", p.Metadata.Name, metadata, p.Spec.PodSelector.MatchLabels)
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

type proxyEnvVar struct {
	Name      string `yaml:"name"`
	ValueFrom *struct {
		SecretKeyRef *struct {
			Name     string `yaml:"name"`
			Key      string `yaml:"key"`
			Optional *bool  `yaml:"optional"`
		} `yaml:"secretKeyRef"`
	} `yaml:"valueFrom"`
}

// litellmProxyManifest returns the proxy's nginx server template and its container's env.
func litellmProxyManifest(t *testing.T) (conf string, env []proxyEnvVar) {
	t.Helper()
	dec := yaml.NewDecoder(strings.NewReader(readFileOrFail(t, filepath.Join("..", "k8s", "agents", "litellm-proxy.yaml"))))
	for {
		var doc struct {
			Kind string            `yaml:"kind"`
			Data map[string]string `yaml:"data"`
			Spec struct {
				Template struct {
					Spec struct {
						Containers []struct {
							Env []proxyEnvVar `yaml:"env"`
						} `yaml:"containers"`
					} `yaml:"spec"`
				} `yaml:"template"`
			} `yaml:"spec"`
		}
		err := dec.Decode(&doc)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("decoding k8s/agents/litellm-proxy.yaml: %v", err)
		}
		switch doc.Kind {
		case "ConfigMap":
			conf = doc.Data["default.conf.template"]
		case "Deployment":
			for _, c := range doc.Spec.Template.Spec.Containers {
				env = append(env, c.Env...)
			}
		}
	}
	if conf == "" {
		t.Fatal("no default.conf.template in the litellm-proxy ConfigMap")
	}
	return conf, env
}

// The proxy relays only the routes the reviewers call, forwards the path it matched, and checks
// the upstream's certificate before it sends the key.
func TestBlisLiteLLMProxyRoutesAndTLS(t *testing.T) {
	conf, _ := litellmProxyManifest(t)
	relays, catchAll := 0, false
	for _, loc := range nginxBlocks(t, conf, "location") {
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

// The proxy relays a request only when it carries the caller token, in x-api-key or as a Bearer
// Authorization, and takes that token from its own secret: a run without the repository secret
// (a fork's) gets a 401 and never reaches the upstream.
func TestBlisLiteLLMProxyRequiresTheCallerToken(t *testing.T) {
	conf, env := litellmProxyManifest(t)
	maps := map[string]string{}
	for _, m := range nginxBlocks(t, conf, "map") {
		maps[m.args] = m.body
	}
	for args, entries := range map[string][]string{
		"$http_x_api_key $litellm_caller_key":                          {"default 0;", `"" 0;`, `"$LITELLM_CALLER_TOKEN" 1;`},
		"$http_authorization $litellm_caller_bearer":                   {"default 0;", `"Bearer " 0;`, `"Bearer $LITELLM_CALLER_TOKEN" 1;`},
		"$litellm_caller_key$litellm_caller_bearer $litellm_caller_ok": {"default 1;", "00 0;"},
	} {
		body, ok := maps[args]
		if !ok {
			t.Errorf("no `map %s` in the proxy config", args)
			continue
		}
		for _, l := range strings.Split(body, "\n") {
			if l = strings.TrimSpace(l); l != "" && !slices.Contains(entries, l) {
				t.Errorf("map %s: unexpected entry %q", args, l)
			}
		}
		for _, e := range entries {
			if !strings.Contains(body, e) {
				t.Errorf("map %s: missing %q", args, e)
			}
		}
	}
	gate := regexp.MustCompile(`if \(\$litellm_caller_ok = 0\) \{\s*return 401;\s*\}`)
	relays := 0
	for _, loc := range nginxBlocks(t, conf, "location") {
		if !strings.Contains(loc.body, "proxy_pass") {
			continue
		}
		relays++
		if !gate.MatchString(loc.body) {
			t.Errorf("location %q relays without checking the caller token (if ($litellm_caller_ok = 0) { return 401; })", loc.args)
		}
	}
	if relays == 0 {
		t.Error("no relaying location found")
	}
	found := false
	for _, e := range env {
		if e.Name != "LITELLM_CALLER_TOKEN" {
			continue
		}
		found = true
		ref := e.ValueFrom
		if ref == nil || ref.SecretKeyRef == nil || ref.SecretKeyRef.Name != "blis-litellm-proxy-token" ||
			ref.SecretKeyRef.Key != "token" || (ref.SecretKeyRef.Optional != nil && *ref.SecretKeyRef.Optional) {
			t.Error("LITELLM_CALLER_TOKEN must come from the secret blis-litellm-proxy-token, key token, not optional")
		}
	}
	if !found {
		t.Error("the proxy container has no LITELLM_CALLER_TOKEN env")
	}
}

type nginxBlock struct{ args, body string }

// nginxBlocks returns each `<directive> <args> { <body> }` in an nginx config template, comments
// dropped and each envsubst ${NAME} read as $NAME. Blocks nested in a body stay in that body.
func nginxBlocks(t *testing.T, conf, directive string) []nginxBlock {
	t.Helper()
	lines := strings.Split(conf, "\n")
	for i, l := range lines {
		if c := strings.Index(l, "#"); c >= 0 {
			lines[i] = l[:c]
		}
	}
	envsubst := regexp.MustCompile(`\$\{(\w+)\}`)
	head := regexp.MustCompile(`(?m)^\s*` + regexp.QuoteMeta(directive) + `\s+([^{;]*)\{`)
	text := envsubst.ReplaceAllString(strings.Join(lines, "\n"), "$$$1")
	var out []nginxBlock
	for _, m := range head.FindAllStringSubmatchIndex(text, -1) {
		depth, i := 1, m[1]
		for ; i < len(text) && depth > 0; i++ {
			switch text[i] {
			case '{':
				depth++
			case '}':
				depth--
			}
		}
		if depth != 0 {
			t.Fatalf("unbalanced %s block at %q", directive, text[m[0]:m[1]])
		}
		out = append(out, nginxBlock{args: strings.TrimSpace(text[m[2]:m[3]]), body: text[m[1] : i-1]})
	}
	return out
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

// The label the policies trust is set by the agent scale sets alone, each on the pool its jobs
// name in runs-on (swapped names would put the untrusted reviewer on the pods that reach LiteLLM),
// and no runner pod gets a service account token (a pod that could read the API could read the
// LiteLLM secret).
func TestScaleSetLabelsAndTokens(t *testing.T) {
	type pool struct{ name, label string }
	want := map[string]pool{
		"agents/values-agents.yaml":    {agentRunsOn, "agents"},
		"agents/values-pr-review.yaml": {"blis-ci-pr-review", "pr-review"},
	}
	files, err := filepath.Glob(filepath.Join("..", "k8s", "ci", "values-*.yaml"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no k8s/ci/values-*.yaml found (%v)", err)
	}
	for _, f := range files {
		want["ci/"+filepath.Base(f)] = pool{}
	}
	for f, w := range want {
		label := w.label
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
		switch got := v.RunnerScaleSetName; {
		case w.name != "" && got != w.name:
			t.Errorf("k8s/%s: runnerScaleSetName is %q, want %q", f, got, w.name)
		case w.name == "" && (got == agentRunsOn || got == "blis-ci-pr-review"):
			t.Errorf("k8s/%s: a CI scale set takes the agent pool name %q", f, got)
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
	// Any LiteLLM secret but the proxy's caller token, which only the reviewer holds
	// (TestBlisProxyTokenOnlyInTheReviewJob).
	agentSecret := regexp.MustCompile(`secrets\.LITELLM_\w+`)
	holds := func(n yaml.Node) bool {
		for _, m := range agentSecret.FindAllString(strings.Join(flattenScalars(n), "\n"), -1) {
			if m != "secrets.LITELLM_PROXY_TOKEN" {
				return true
			}
		}
		return false
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

// The proxy's caller token goes to pr-review.yml's review job alone: any other job holding it
// could use the proxy from a pr-review pod, and a workflow-level reference reaches every job.
func TestBlisProxyTokenOnlyInTheReviewJob(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("..", ".github", "workflows", "*.yml"))
	if err != nil {
		t.Fatal(err)
	}
	const token = "secrets.LITELLM_PROXY_TOKEN"
	refs := 0
	for _, f := range files {
		var wf map[string]yaml.Node
		body, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if err := yaml.Unmarshal(body, &wf); err != nil {
			t.Fatalf("parsing %s: %v", f, err)
		}
		name := filepath.Base(f)
		for key, n := range wf {
			if key != "jobs" && strings.Contains(strings.Join(flattenScalars(n), "\n"), token) {
				t.Errorf("%s references %s at workflow level (%s:)", name, token, key)
			}
		}
		var jobs map[string]yaml.Node
		if n, ok := wf["jobs"]; ok {
			if err := n.Decode(&jobs); err != nil {
				t.Fatalf("parsing %s jobs: %v", f, err)
			}
		}
		for job, n := range jobs {
			if !strings.Contains(strings.Join(flattenScalars(n), "\n"), token) {
				continue
			}
			if name != "pr-review.yml" || job != "review" {
				t.Errorf("%s job %q references %s, which is pr-review.yml's review job's alone", name, job, token)
				continue
			}
			refs++
		}
	}
	if refs == 0 {
		t.Errorf("pr-review.yml's review job does not send %s, so the proxy refuses it", token)
	}
}
