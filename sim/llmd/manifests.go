package llmd

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"sort"
	"strings"

	apixv1 "github.com/llm-d/llm-d-router/apix/v1"
	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	giev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	sigsyaml "sigs.k8s.io/yaml"
)

// newScheme registers every resource kind a router deployment's manifests can carry.
func newScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(s))
	utilruntime.Must(giev1.Install(s))
	s.AddKnownTypes(schema.GroupVersion(apixv1.GroupVersion), &apixv1.InferenceObjective{}, &apixv1.InferenceObjectiveList{})
	metav1.AddToGroupVersion(s, schema.GroupVersion(apixv1.GroupVersion))
	s.AddKnownTypes(schema.GroupVersion(v1alpha2.GroupVersion),
		&v1alpha2.InferenceObjective{}, &v1alpha2.InferenceObjectiveList{},
		&v1alpha2.InferenceModelRewrite{}, &v1alpha2.InferenceModelRewriteList{})
	metav1.AddToGroupVersion(s, schema.GroupVersion(v1alpha2.GroupVersion))
	return s
}

// resources are the router-relevant objects of a deployment's manifests.
type resources struct {
	nodes          []*corev1.Node
	pools          []*giev1.InferencePool
	objectives     []*apixv1.InferenceObjective
	objectivesV1a2 []*v1alpha2.InferenceObjective
	rewrites       []*v1alpha2.InferenceModelRewrite
	// served are the custom resource kinds and versions the simulated cluster's CRDs serve.
	served map[schema.GroupVersionKind]bool
}

func (r *resources) objects() []client.Object {
	var out []client.Object
	for _, o := range r.nodes {
		out = append(out, o)
	}
	for _, o := range r.pools {
		out = append(out, o)
	}
	for _, o := range r.objectives {
		out = append(out, o)
	}
	for _, o := range r.objectivesV1a2 {
		out = append(out, o)
	}
	for _, o := range r.rewrites {
		out = append(out, o)
	}
	return out
}

// parseManifests decodes multi-document Kubernetes YAML. BLIS retains Node records for
// declared topology; kinds neither BLIS nor the router reads (the model servers' own
// Deployments, Services, ConfigMaps, ...) are accepted and ignored. Unknown group-versions
// are an error. Custom
// resources get their CRD defaults, as on admission to an API server.
func parseManifests(scheme *runtime.Scheme, docs []byte, namespace string) (*resources, error) {
	decoder := serializer.NewCodecFactory(scheme, serializer.EnableStrict).UniversalDeserializer() // unknown or duplicate fields are errors, as with strict field validation on admission
	reader := k8syaml.NewYAMLReader(bufio.NewReader(bytes.NewReader(docs)))
	defaults, err := newCRDDefaulter()
	if err != nil {
		return nil, err
	}
	res := &resources{served: map[schema.GroupVersionKind]bool{}}
	for gvk := range defaults {
		res.served[gvk] = true
	}
	for i := 0; ; i++ {
		doc, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("manifest document %d: %w", i, err)
		}
		if len(bytes.TrimSpace(doc)) == 0 {
			continue
		}
		if doc, err = defaults.apply(doc); err != nil {
			return nil, fmt.Errorf("manifest document %d: %w", i, err)
		}
		obj, gvk, err := decoder.Decode(doc, nil, nil)
		if err != nil {
			if runtime.IsNotRegisteredError(err) {
				var tm metav1.TypeMeta
				_ = sigsyaml.Unmarshal(doc, &tm)
				if gv, _ := schema.ParseGroupVersion(tm.APIVersion); routerGroups[gv.Group] {
					return nil, fmt.Errorf("manifest document %d: %s %s is not served by the router's CRDs (%s)", i, tm.APIVersion, tm.Kind, servedList(res.served))
				}
				continue // a kind the router never reads
			}
			return nil, fmt.Errorf("manifest document %d: %w", i, err)
		}
		if mo, ok := obj.(metav1.Object); ok && mo.GetNamespace() == "" {
			if _, isNode := obj.(*corev1.Node); !isNode {
				mo.SetNamespace(namespace)
			}
		}
		switch o := obj.(type) {
		case *corev1.Node:
			res.nodes = append(res.nodes, o)
		case *giev1.InferencePool:
			res.pools = append(res.pools, o)
		case *apixv1.InferenceObjective:
			res.objectives = append(res.objectives, o)
		case *v1alpha2.InferenceObjective:
			res.objectivesV1a2 = append(res.objectivesV1a2, o)
		case *v1alpha2.InferenceModelRewrite:
			res.rewrites = append(res.rewrites, o)
		case *corev1.Pod:
			return nil, fmt.Errorf("manifest document %d: a Pod manifest cannot be simulated: BLIS creates the model-server pods from the scenario", i)
		default:
			_ = gvk // other kinds carry no router configuration
		}
	}
	return res, nil
}

// routerGroups are the API groups of the router's resources. A manifest in one of them that no
// installed CRD serves is an error, as applying it to a cluster would be.
var routerGroups = map[string]bool{"llm-d.ai": true, "inference.networking.k8s.io": true, "inference.networking.x-k8s.io": true}

func servedList(served map[schema.GroupVersionKind]bool) string {
	var out []string
	for gvk := range served {
		out = append(out, gvk.GroupVersion().String()+" "+gvk.Kind)
	}
	sort.Strings(out)
	return strings.Join(out, ", ")
}
