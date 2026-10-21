package llmd

import (
	"embed"
	"fmt"
	"path"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/defaulting"
	"k8s.io/apimachinery/pkg/runtime/schema"
	k8syaml "sigs.k8s.io/yaml"
)

// crdFiles are the CustomResourceDefinitions of the router's resources, copied verbatim from the
// modules BLIS builds against (llm-d-router and gateway-api-inference-extension, config/crd/bases);
// TestCRDsMatchModules keeps the copies identical to the modules' files.
//
//go:embed crds/*.yaml
var crdFiles embed.FS

// crdDefaulter applies the defaults the Kubernetes API server applies to a custom resource on
// admission, from the resource's own CRD schema. A real cluster's router reads the defaulted
// objects (for example an InferenceObjective poolRef's group), so the simulated cluster does too.
type crdDefaulter map[schema.GroupVersionKind]*structuralschema.Structural

func newCRDDefaulter() (crdDefaulter, error) {
	entries, err := crdFiles.ReadDir("crds")
	if err != nil {
		return nil, err
	}
	d := crdDefaulter{}
	for _, e := range entries {
		b, err := crdFiles.ReadFile(path.Join("crds", e.Name()))
		if err != nil {
			return nil, err
		}
		var crd apiextensionsv1.CustomResourceDefinition
		if err := k8syaml.UnmarshalStrict(b, &crd); err != nil {
			return nil, fmt.Errorf("CRD %s: %w", e.Name(), err)
		}
		for _, v := range crd.Spec.Versions {
			if !v.Served || v.Schema == nil || v.Schema.OpenAPIV3Schema == nil {
				continue
			}
			var internal apiextensions.JSONSchemaProps
			if err := apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(v.Schema.OpenAPIV3Schema, &internal, nil); err != nil {
				return nil, fmt.Errorf("CRD %s %s: %w", e.Name(), v.Name, err)
			}
			s, err := structuralschema.NewStructural(&internal)
			if err != nil {
				return nil, fmt.Errorf("CRD %s %s: %w", e.Name(), v.Name, err)
			}
			d[schema.GroupVersionKind{Group: crd.Spec.Group, Version: v.Name, Kind: crd.Spec.Names.Kind}] = s
		}
	}
	return d, nil
}

// apply defaults one manifest document, if it is a custom resource the defaulter knows; other
// documents are returned unchanged.
func (d crdDefaulter) apply(doc []byte) ([]byte, error) {
	var obj map[string]any
	if err := k8syaml.Unmarshal(doc, &obj); err != nil {
		return nil, err
	}
	apiVersion, _ := obj["apiVersion"].(string)
	kind, _ := obj["kind"].(string)
	gv, err := schema.ParseGroupVersion(apiVersion)
	if err != nil {
		return doc, nil // the decoder reports it
	}
	s, ok := d[gv.WithKind(kind)]
	if !ok {
		return doc, nil
	}
	defaulting.Default(obj, s)
	return k8syaml.Marshal(obj)
}
