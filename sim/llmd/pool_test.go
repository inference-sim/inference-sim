package llmd

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// poolManifests is a router deployment's InferencePool with two InferenceObjectives: requests
// naming "critical" are protected, requests naming "sheddable" may be shed under saturation.
const poolManifests = `apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: vllm
spec:
  selector:
    matchLabels:
      app: vllm
  targetPorts:
  - number: 8000
  endpointPickerRef:
    name: epp
    port:
      number: 9002
---
apiVersion: llm-d.ai/v1
kind: InferenceObjective
metadata:
  name: critical
spec:
  priority: 4
  poolRefs:
  - name: vllm
---
apiVersion: llm-d.ai/v1
kind: InferenceObjective
metadata:
  name: sheddable
spec:
  priority: -2
  poolRefs:
  - name: vllm
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
spec:
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai
`

func runPool(t *testing.T, manifests string, labels map[string]string) runResult {
	t.Helper()
	reqs := testRequests(200, 3, 2000, 0) // 500 req/s over 400 ms: the pool saturates across scrapes
	for i, r := range reqs {
		r.SLOClass = map[bool]string{true: "critical", false: "sheddable"}[i%2 == 0]
	}
	dep := testDeployment(2)
	dep.BatchConfig = sim.NewBatchConfig(4, 2048, 0)
	cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{
		Model:     "test-model",
		EPPFlags:  []string{"--pool-name=vllm"},
		Manifests: []byte(manifests),
		PodLabels: func(cluster.InstanceID, cluster.PoolRole) map[string]string { return labels },
	})
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	return runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}
}

// TestPool_ManifestObjectivesGovernAdmission: in --pool-name mode the router reconciles the
// deployment's InferencePool and InferenceObjectives. Its legacy admission sheds only requests
// whose objective has negative priority.
func TestPool_ManifestObjectivesGovernAdmission(t *testing.T) {
	res := runPool(t, poolManifests, map[string]string{"app": "vllm"})
	assertConservation(t, res, len(res.reqs))
	shed := map[string]int{}
	for _, r := range res.reqs {
		if r.AssignedInstance == "" {
			shed[r.SLOClass]++
		}
	}
	if shed["critical"] != 0 {
		t.Errorf("%d critical (priority 4) requests were not dispatched", shed["critical"])
	}
	if res.cs.RejectedRequests() == 0 || shed["sheddable"] != res.cs.RejectedRequests() {
		t.Errorf("admission rejected %d requests, %d of them sheddable; want the saturated pool to shed sheddable requests only",
			res.cs.RejectedRequests(), shed["sheddable"])
	}
	t.Logf("shed %d sheddable of %d", shed["sheddable"], len(res.reqs)/2)
}

// TestPool_SelectorMustMatchPods: the InferencePool selects pods by label, as in a cluster; pods
// without the labels are not endpoints, so nothing is served.
func TestPool_SelectorMustMatchPods(t *testing.T) {
	res := runPool(t, poolManifests, nil)
	if res.metrics.CompletedRequests != 0 {
		t.Errorf("%d requests completed on pods the InferencePool selector does not match", res.metrics.CompletedRequests)
	}
	if res.cs.RoutingRejections() == 0 {
		t.Errorf("no routing rejection with an empty pool")
	}
	assertConservation(t, res, len(res.reqs))
}

// TestPool_V1Alpha2ObjectivesReconciled: llm-d.ai/v1alpha2 objectives (the shape of llm-d's own
// sample manifests) are served by the installed CRD and reconciled alongside v1, as the runner
// does when discovery finds both versions.
func TestPool_V1Alpha2ObjectivesReconciled(t *testing.T) {
	manifests := strings.Replace(poolManifests, `apiVersion: llm-d.ai/v1
kind: InferenceObjective
metadata:
  name: sheddable
spec:
  priority: -2
  poolRefs:
  - name: vllm`, `apiVersion: llm-d.ai/v1alpha2
kind: InferenceObjective
metadata:
  name: sheddable
spec:
  priority: -2
  poolRef:
    name: vllm`, 1)
	if manifests == poolManifests {
		t.Fatal("fixture substitution did not apply")
	}
	res := runPool(t, manifests, map[string]string{"app": "vllm"})
	var prios []string
	for _, o := range res.gw.emb.Datastore.ObjectiveGetAll() {
		prios = append(prios, fmt.Sprintf("%s=%d", o.Name, *o.Spec.Priority))
	}
	sort.Strings(prios)
	if got := strings.Join(prios, ","); got != "critical=4,sheddable=-2" {
		t.Errorf("reconciled objectives %s, want critical=4,sheddable=-2", got)
	}
}

// TestParseManifests_UnservedRouterVersionRefused: a manifest in a router API group at a version
// no installed CRD serves is an error, as applying it to a cluster is; other unknown kinds are
// ignored.
func TestParseManifests_UnservedRouterVersionRefused(t *testing.T) {
	_, err := parseManifests(newScheme(), []byte("apiVersion: llm-d.ai/v9\nkind: InferenceObjective\nmetadata:\n  name: x\n"), defaultNamespace)
	if err == nil || !strings.Contains(err.Error(), "not served") {
		t.Errorf("err = %v, want a refusal naming the unserved version", err)
	}
	if _, err := parseManifests(newScheme(), []byte("apiVersion: example.com/v1\nkind: Widget\nmetadata:\n  name: x\n"), defaultNamespace); err != nil {
		t.Errorf("a kind the router never reads must be ignored: %v", err)
	}
}
