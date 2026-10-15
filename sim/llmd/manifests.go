package llmd

import (
	"bufio"
	"bytes"
	"fmt"
	"io"

	apixv1 "github.com/llm-d/llm-d-router/apix/v1"
	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	giev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// newScheme registers every resource kind a router deployment's manifests can carry.
func newScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(s))
	utilruntime.Must(giev1.Install(s))
	s.AddKnownTypes(apixv1.SchemeGroupVersion, &apixv1.InferenceObjective{}, &apixv1.InferenceObjectiveList{})
	metav1.AddToGroupVersion(s, apixv1.SchemeGroupVersion)
	s.AddKnownTypes(v1alpha2.SchemeGroupVersion,
		&v1alpha2.InferenceObjective{}, &v1alpha2.InferenceObjectiveList{},
		&v1alpha2.InferenceModelRewrite{}, &v1alpha2.InferenceModelRewriteList{})
	metav1.AddToGroupVersion(s, v1alpha2.SchemeGroupVersion)
	return s
}

// resources are the router-relevant objects of a deployment's manifests.
type resources struct {
	pools          []*giev1.InferencePool
	objectives     []*apixv1.InferenceObjective
	objectivesV1a2 []*v1alpha2.InferenceObjective
	rewrites       []*v1alpha2.InferenceModelRewrite
}

func (r *resources) objects() []client.Object {
	var out []client.Object
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

// parseManifests decodes multi-document Kubernetes YAML. Kinds the router does not read (the
// model servers' own Deployments, Services, ConfigMaps, ...) are accepted and ignored, because a
// real deployment's manifest set carries them too; unknown group-versions are an error.
func parseManifests(scheme *runtime.Scheme, docs []byte, namespace string) (*resources, error) {
	decoder := serializer.NewCodecFactory(scheme).UniversalDeserializer()
	reader := k8syaml.NewYAMLReader(bufio.NewReader(bytes.NewReader(docs)))
	res := &resources{}
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
		obj, gvk, err := decoder.Decode(doc, nil, nil)
		if err != nil {
			if runtime.IsNotRegisteredError(err) {
				continue // a kind the router never reads
			}
			return nil, fmt.Errorf("manifest document %d: %w", i, err)
		}
		if mo, ok := obj.(metav1.Object); ok && mo.GetNamespace() == "" {
			mo.SetNamespace(namespace)
		}
		switch o := obj.(type) {
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

// defaultObjectives are the InferenceObjectives BLIS installs when the deployment supplies no
// manifests: one per BLIS SLO class, at the priorities of the Gateway API Inference Extension
// defaults.
func defaultObjectives(poolNamespace, poolName string) []*apixv1.InferenceObjective {
	prio := map[string]int32{"critical": 4, "standard": 3, "batch": -1, "sheddable": -2, "background": -3}
	var out []*apixv1.InferenceObjective
	for _, name := range []string{"background", "batch", "critical", "sheddable", "standard"} {
		p := prio[name]
		out = append(out, &apixv1.InferenceObjective{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: poolNamespace},
			Spec: apixv1.InferenceObjectiveSpec{
				Priority: &p,
				PoolRefs: []apixv1.PoolObjectReference{{Name: apixv1.ObjectName(poolName), Group: giev1.GroupName, Kind: "InferencePool"}},
			},
		})
	}
	return out
}
