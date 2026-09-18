package sim

import "testing"

func TestNeverSaturated_AlwaysReturnsZero(t *testing.T) {
	detector := &NeverSaturated{}

	tests := []struct {
		name  string
		state *RouterState
	}{
		{"nil snapshots", &RouterState{}},
		{"empty snapshots", &RouterState{Snapshots: []RoutingSnapshot{}}},
		{"loaded cluster", &RouterState{Snapshots: []RoutingSnapshot{
			{ID: "i0", QueueDepth: 100, KVUtilization: 0.99, InFlightRequests: 50},
			{ID: "i1", QueueDepth: 100, KVUtilization: 0.99, InFlightRequests: 50},
		}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sat := detector.Saturation(tt.state)
			if sat != 0.0 {
				t.Errorf("NeverSaturated.Saturation() = %f, want 0.0", sat)
			}
		})
	}
}

func TestUtilizationDetector_Saturation(t *testing.T) {
	tests := []struct {
		name      string
		queueThr  float64
		kvThr     float64
		snapshots []RoutingSnapshot
		wantMin   float64
		wantMax   float64
	}{
		{
			name:      "empty snapshots returns 1.0",
			queueThr:  5,
			kvThr:     0.8,
			snapshots: nil,
			wantMin:   1.0,
			wantMax:   1.0,
		},
		{
			name:     "below thresholds",
			queueThr: 10,
			kvThr:    0.8,
			snapshots: []RoutingSnapshot{
				{ID: "i0", QueueDepth: 2, KVUtilization: 0.3},
				{ID: "i1", QueueDepth: 3, KVUtilization: 0.4},
			},
			wantMin: 0.0,
			wantMax: 0.5, // avg(max(2/10,0.3/0.8), max(3/10,0.4/0.8)) = avg(0.375, 0.5) = 0.4375
		},
		{
			name:     "above thresholds",
			queueThr: 5,
			kvThr:    0.8,
			snapshots: []RoutingSnapshot{
				{ID: "i0", QueueDepth: 6, KVUtilization: 0.9},
				{ID: "i1", QueueDepth: 7, KVUtilization: 0.95},
			},
			wantMin: 1.0,
			wantMax: 2.0, // both instances saturated
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewUtilizationDetector(tt.queueThr, tt.kvThr)
			state := &RouterState{Snapshots: tt.snapshots}
			sat := d.Saturation(state)
			if sat < tt.wantMin || sat > tt.wantMax {
				t.Errorf("Saturation() = %f, want in [%f, %f]", sat, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestConcurrencyDetector_Saturation(t *testing.T) {
	tests := []struct {
		name           string
		maxConcurrency int
		snapshots      []RoutingSnapshot
		wantMin        float64
		wantMax        float64
	}{
		{
			name:           "empty snapshots returns 1.0",
			maxConcurrency: 100,
			snapshots:      nil,
			wantMin:        1.0,
			wantMax:        1.0,
		},
		{
			name:           "below capacity",
			maxConcurrency: 100,
			snapshots: []RoutingSnapshot{
				{ID: "i0", InFlightRequests: 30},
				{ID: "i1", InFlightRequests: 20},
			},
			wantMin: 0.24,
			wantMax: 0.26, // 50 / (2*100) = 0.25
		},
		{
			name:           "at capacity",
			maxConcurrency: 50,
			snapshots: []RoutingSnapshot{
				{ID: "i0", InFlightRequests: 50},
				{ID: "i1", InFlightRequests: 50},
			},
			wantMin: 1.0,
			wantMax: 1.0, // 100 / (2*50) = 1.0
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewConcurrencyDetector(tt.maxConcurrency)
			state := &RouterState{Snapshots: tt.snapshots}
			sat := d.Saturation(state)
			if sat < tt.wantMin || sat > tt.wantMax {
				t.Errorf("Saturation() = %f, want in [%f, %f]", sat, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestUtilizationDetector_InvalidParams_Panics(t *testing.T) {
	tests := []struct {
		name     string
		queueThr float64
		kvThr    float64
	}{
		{"zero queue threshold", 0, 0.8},
		{"negative kv threshold", 5, -0.1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				}
			}()
			NewUtilizationDetector(tt.queueThr, tt.kvThr)
		})
	}
}

func TestConcurrencyDetector_InvalidParams_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for maxConcurrency <= 0")
		}
	}()
	NewConcurrencyDetector(0)
}

func TestNewSaturationDetector_Factory(t *testing.T) {
	tests := []struct {
		name     string
		detector string
		wantZero bool
	}{
		{"never", "never", true},
		{"empty string", "", true},
	}
	state := &RouterState{Snapshots: []RoutingSnapshot{{ID: "i0", QueueDepth: 10, KVUtilization: 0.9, InFlightRequests: 50}}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewSaturationDetector(tt.detector, 5, 0.8, 100)
			sat := d.Saturation(state)
			if tt.wantZero && sat != 0.0 {
				t.Errorf("expected 0.0, got %f", sat)
			}
		})
	}
}

func TestNewSaturationDetector_Factory_UtilizationAndConcurrency(t *testing.T) {
	state := &RouterState{Snapshots: []RoutingSnapshot{{ID: "i0", QueueDepth: 10, KVUtilization: 0.9, InFlightRequests: 50}}}

	uDet := NewSaturationDetector("utilization", 5, 0.8, 100)
	uSat := uDet.Saturation(state)
	if uSat <= 0 {
		t.Errorf("utilization detector should return > 0 for loaded state, got %f", uSat)
	}

	cDet := NewSaturationDetector("concurrency", 5, 0.8, 100)
	cSat := cDet.Saturation(state)
	if cSat <= 0 {
		t.Errorf("concurrency detector should return > 0 for loaded state, got %f", cSat)
	}
}

func TestNewSaturationDetector_Factory_UnknownPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unknown detector")
		}
	}()
	NewSaturationDetector("bogus", 5, 0.8, 100)
}

func TestIsValidSaturationDetector(t *testing.T) {
	for _, name := range []string{"", "never", "utilization", "concurrency"} {
		if !IsValidSaturationDetector(name) {
			t.Errorf("expected %q to be valid", name)
		}
	}
	if IsValidSaturationDetector("bogus") {
		t.Error("expected 'bogus' to be invalid")
	}
}
