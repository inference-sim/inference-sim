package embed

import (
	"errors"
	"fmt"
	"testing"
)

// TestPoolIDMatchesKubernetesForm: a host keys per-router state (for example a random-stream
// scope) on the pool's identity, so the rendering is part of the contract. It is the form
// Kubernetes gives a GroupKind and a NamespacedName, "Kind.group namespace/name", with the
// group and its dot left out when the group is empty (the core group).
func TestPoolIDMatchesKubernetesForm(t *testing.T) {
	for _, c := range []struct {
		pool Pool
		want string
	}{
		{Pool{Namespace: "default", Name: "epp", Group: "apps", Kind: "Deployment"}, "Deployment.apps default/epp"},
		{Pool{Namespace: "ns", Name: "pool", Group: "inference.networking.k8s.io", Kind: "InferencePool"},
			"InferencePool.inference.networking.k8s.io ns/pool"},
		{Pool{Namespace: "ns", Name: "pool", Kind: "InferencePool"}, "InferencePool ns/pool"},
	} {
		if got := c.pool.ID(); got != c.want {
			t.Errorf("%+v: ID() = %q, want %q", c.pool, got, c.want)
		}
	}
}

// TestEndpointAddressIsHostPort: the router reports endpoints by address and the host maps
// them back, so both sides must render an address alike, IPv6 included.
func TestEndpointAddressIsHostPort(t *testing.T) {
	for _, c := range []struct {
		ep   Endpoint
		want string
	}{
		{Endpoint{IP: "10.0.0.1", Port: 8000}, "10.0.0.1:8000"},
		{Endpoint{IP: "fd00::1", Port: 8000}, "[fd00::1]:8000"},
	} {
		if got := c.ep.Address(); got != c.want {
			t.Errorf("%+v: Address() = %q, want %q", c.ep, got, c.want)
		}
	}
}

// TestRefusalErrorSurvivesWrapping: a host words its own refusal message, so it must be able
// to recover the plugin, type and reason from whatever error the constructor wraps them in.
func TestRefusalErrorSurvivesWrapping(t *testing.T) {
	err := fmt.Errorf("building the router: %w", &RefusalError{Plugin: "p", Type: "t", Reason: "r"})
	var re *RefusalError
	if !errors.As(err, &re) {
		t.Fatalf("errors.As did not find the RefusalError in %v", err)
	}
	if re.Plugin != "p" || re.Type != "t" || re.Reason != "r" {
		t.Errorf("recovered %+v", re)
	}
	if got, want := re.Error(), `plugin "p" (type t): r`; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

// TestKVEventsAreTheTwoBlockEvents: KVEvent is closed to this package, so a router's type
// switch over the two kinds is exhaustive within a major version.
func TestKVEventsAreTheTwoBlockEvents(t *testing.T) {
	events := []KVEvent{BlockStored{}, BlockRemoved{}}
	for _, e := range events {
		switch e.(type) {
		case BlockStored, BlockRemoved:
		default:
			t.Errorf("unexpected event type %T", e)
		}
	}
}
