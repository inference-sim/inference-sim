package cluster

import "testing"

// mustCreatePool creates a ReplicaPool or fails the test
func mustCreatePool(t *testing.T, poolID string, poolType PoolType, min, max int) *ReplicaPool {
	t.Helper()
	pool, err := NewReplicaPool(poolID, poolType, min, max)
	if err != nil {
		t.Fatalf("NewReplicaPool(%q, %v, %d, %d) failed: %v", poolID, poolType, min, max, err)
	}
	return pool
}

// TestNewReplicaPool tests ReplicaPool creation
func TestNewReplicaPool(t *testing.T) {
	tests := []struct {
		name        string
		poolID      string
		poolType    PoolType
		minReplicas int
		maxReplicas int
		wantError   bool
		errorMsg    string
	}{
		{
			name:        "Valid pool",
			poolID:      "pool1",
			poolType:    PoolMonolithic,
			minReplicas: 1,
			maxReplicas: 5,
			wantError:   false,
		},
		{
			name:        "Min equals max",
			poolID:      "pool2",
			poolType:    PoolMonolithic,
			minReplicas: 3,
			maxReplicas: 3,
			wantError:   false,
		},
		{
			name:        "Min = 0",
			poolID:      "pool3",
			poolType:    PoolMonolithic,
			minReplicas: 0,
			maxReplicas: 5,
			wantError:   false,
		},
		{
			name:        "Invalid: MinReplicas < 0",
			poolID:      "pool4",
			poolType:    PoolMonolithic,
			minReplicas: -1,
			maxReplicas: 5,
			wantError:   true,
			errorMsg:    "MinReplicas must be >= 0",
		},
		{
			name:        "Invalid: MaxReplicas < MinReplicas",
			poolID:      "pool5",
			poolType:    PoolMonolithic,
			minReplicas: 5,
			maxReplicas: 3,
			wantError:   true,
			errorMsg:    "MaxReplicas (3) must be >= MinReplicas (5)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pool, err := NewReplicaPool(tt.poolID, tt.poolType, tt.minReplicas, tt.maxReplicas)
			if tt.wantError {
				if err == nil {
					t.Errorf("NewReplicaPool() expected error containing '%s', got nil", tt.errorMsg)
				} else if !contains(err.Error(), tt.errorMsg) {
					t.Errorf("NewReplicaPool() error = '%v', want error containing '%s'", err, tt.errorMsg)
				}
			} else {
				if err != nil {
					t.Errorf("NewReplicaPool() unexpected error: %v", err)
				}
				if pool == nil {
					t.Error("NewReplicaPool() returned nil pool")
				}
				if pool.PoolID != tt.poolID {
					t.Errorf("PoolID = %s, want %s", pool.PoolID, tt.poolID)
				}
				if pool.MinReplicas != tt.minReplicas {
					t.Errorf("MinReplicas = %d, want %d", pool.MinReplicas, tt.minReplicas)
				}
				if pool.MaxReplicas != tt.maxReplicas {
					t.Errorf("MaxReplicas = %d, want %d", pool.MaxReplicas, tt.maxReplicas)
				}
				if pool.Len() != 0 {
					t.Errorf("Initial pool size = %d, want 0", pool.Len())
				}
			}
		})
	}
}

// TestReplicaPool_AddInstance tests adding instances within bounds
func TestReplicaPool_AddInstance(t *testing.T) {
	t.Run("Add instance within bounds", func(t *testing.T) {
		pool := mustCreatePool(t, "pool1", PoolMonolithic, 1, 3)

		inst1 := &InstanceSimulator{ID: "inst1"}
		inst2 := &InstanceSimulator{ID: "inst2"}

		if err := pool.AddInstance(inst1); err != nil {
			t.Errorf("AddInstance(inst1) unexpected error: %v", err)
		}
		if pool.Len() != 1 {
			t.Errorf("After adding inst1, pool size = %d, want 1", pool.Len())
		}

		if err := pool.AddInstance(inst2); err != nil {
			t.Errorf("AddInstance(inst2) unexpected error: %v", err)
		}
		if pool.Len() != 2 {
			t.Errorf("After adding inst2, pool size = %d, want 2", pool.Len())
		}
	})

	t.Run("Add instance exceeds max", func(t *testing.T) {
		pool := mustCreatePool(t, "pool2", PoolMonolithic, 1, 2)

		inst1 := &InstanceSimulator{ID: "inst1"}
		inst2 := &InstanceSimulator{ID: "inst2"}
		inst3 := &InstanceSimulator{ID: "inst3"}

		if err := pool.AddInstance(inst1); err != nil {
			t.Fatalf("AddInstance(inst1) failed: %v", err)
		}
		if err := pool.AddInstance(inst2); err != nil {
			t.Fatalf("AddInstance(inst2) failed: %v", err)
		}

		err := pool.AddInstance(inst3)
		if err == nil {
			t.Error("AddInstance() expected error when exceeding MaxReplicas, got nil")
		}
		if !contains(err.Error(), "pool at MaxReplicas") {
			t.Errorf("AddInstance() error = '%v', want error containing 'pool at MaxReplicas'", err)
		}
		if pool.Len() != 2 {
			t.Errorf("After failed add, pool size = %d, want 2", pool.Len())
		}
	})
}

// TestReplicaPool_RemoveInstance tests removing instances within bounds
func TestReplicaPool_RemoveInstance(t *testing.T) {
	t.Run("Remove instance within bounds", func(t *testing.T) {
		pool := mustCreatePool(t, "pool1", PoolMonolithic, 1, 3)

		inst1 := &InstanceSimulator{ID: "inst1"}
		inst2 := &InstanceSimulator{ID: "inst2"}

		if err := pool.AddInstance(inst1); err != nil {
			t.Fatalf("AddInstance(inst1) failed: %v", err)
		}
		if err := pool.AddInstance(inst2); err != nil {
			t.Fatalf("AddInstance(inst2) failed: %v", err)
		}

		if err := pool.RemoveInstance("inst2"); err != nil {
			t.Errorf("RemoveInstance(inst2) unexpected error: %v", err)
		}
		if pool.Len() != 1 {
			t.Errorf("After removing inst2, pool size = %d, want 1", pool.Len())
		}

		// Verify inst2 is gone and inst1 remains
		if pool.GetInstance("inst2") != nil {
			t.Error("inst2 should be removed from pool")
		}
		if pool.GetInstance("inst1") == nil {
			t.Error("inst1 should remain in pool")
		}
	})

	t.Run("Remove instance violates min", func(t *testing.T) {
		pool := mustCreatePool(t, "pool2", PoolMonolithic, 1, 3)

		inst1 := &InstanceSimulator{ID: "inst1"}
		if err := pool.AddInstance(inst1); err != nil {
			t.Fatalf("AddInstance(inst1) failed: %v", err)
		}

		err := pool.RemoveInstance("inst1")
		if err == nil {
			t.Error("RemoveInstance() expected error when violating MinReplicas, got nil")
		}
		if !contains(err.Error(), "pool at MinReplicas") {
			t.Errorf("RemoveInstance() error = '%v', want error containing 'pool at MinReplicas'", err)
		}
		if pool.Len() != 1 {
			t.Errorf("After failed remove, pool size = %d, want 1", pool.Len())
		}
	})

	t.Run("Remove non-existent instance", func(t *testing.T) {
		pool := mustCreatePool(t, "pool3", PoolMonolithic, 0, 3)

		inst1 := &InstanceSimulator{ID: "inst1"}
		inst2 := &InstanceSimulator{ID: "inst2"}
		if err := pool.AddInstance(inst1); err != nil {
			t.Fatalf("AddInstance(inst1) failed: %v", err)
		}
		if err := pool.AddInstance(inst2); err != nil {
			t.Fatalf("AddInstance(inst2) failed: %v", err)
		}

		err := pool.RemoveInstance("inst3")
		if err == nil {
			t.Error("RemoveInstance() expected error for non-existent instance, got nil")
		}
		if !contains(err.Error(), "not found") {
			t.Errorf("RemoveInstance() error = '%v', want error containing 'not found'", err)
		}
		if pool.Len() != 2 {
			t.Errorf("After failed remove, pool size = %d, want 2", pool.Len())
		}
	})
}

// TestReplicaPool_GetInstance tests instance retrieval
func TestReplicaPool_GetInstance(t *testing.T) {
	pool := mustCreatePool(t, "pool1", PoolMonolithic, 0, 3)

	inst1 := &InstanceSimulator{ID: "inst1"}
	inst2 := &InstanceSimulator{ID: "inst2"}

	if err := pool.AddInstance(inst1); err != nil {
		t.Fatalf("AddInstance(inst1) failed: %v", err)
	}
	if err := pool.AddInstance(inst2); err != nil {
		t.Fatalf("AddInstance(inst2) failed: %v", err)
	}

	// Test retrieving existing instances
	retrieved1 := pool.GetInstance("inst1")
	if retrieved1 == nil {
		t.Error("GetInstance(inst1) returned nil")
	} else if retrieved1.ID != "inst1" {
		t.Errorf("GetInstance(inst1) returned instance with ID %s", retrieved1.ID)
	}

	retrieved2 := pool.GetInstance("inst2")
	if retrieved2 == nil {
		t.Error("GetInstance(inst2) returned nil")
	} else if retrieved2.ID != "inst2" {
		t.Errorf("GetInstance(inst2) returned instance with ID %s", retrieved2.ID)
	}

	// Test retrieving non-existent instance
	retrieved3 := pool.GetInstance("inst3")
	if retrieved3 != nil {
		t.Error("GetInstance(inst3) should return nil for non-existent instance")
	}
}

// TestReplicaPool_Len tests pool size tracking
func TestReplicaPool_Len(t *testing.T) {
	pool := mustCreatePool(t, "pool1", PoolMonolithic, 0, 5)

	if pool.Len() != 0 {
		t.Errorf("Initial pool size = %d, want 0", pool.Len())
	}

	inst1 := &InstanceSimulator{ID: "inst1"}
	inst2 := &InstanceSimulator{ID: "inst2"}
	inst3 := &InstanceSimulator{ID: "inst3"}

	if err := pool.AddInstance(inst1); err != nil {
		t.Fatalf("AddInstance(inst1) failed: %v", err)
	}
	if pool.Len() != 1 {
		t.Errorf("After 1 add, pool size = %d, want 1", pool.Len())
	}

	if err := pool.AddInstance(inst2); err != nil {
		t.Fatalf("AddInstance(inst2) failed: %v", err)
	}
	if err := pool.AddInstance(inst3); err != nil {
		t.Fatalf("AddInstance(inst3) failed: %v", err)
	}
	if pool.Len() != 3 {
		t.Errorf("After 3 adds, pool size = %d, want 3", pool.Len())
	}

	pool.RemoveInstance("inst2")
	if pool.Len() != 2 {
		t.Errorf("After 1 remove, pool size = %d, want 2", pool.Len())
	}
}

// TestNewReplicaPool_InvalidConfig tests error handling
func TestNewReplicaPool_InvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		min  int
		max  int
	}{
		{"negative min", -1, 5},
		{"max less than min", 5, 3},
		{"both negative", -2, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewReplicaPool("pool1", PoolMonolithic, tt.min, tt.max)
			if err == nil {
				t.Errorf("NewReplicaPool(%d, %d) should have returned error", tt.min, tt.max)
			}
		})
	}
}

// TestReplicaPool_BoundsInvariant tests BC-3: pool bounds invariant
func TestReplicaPool_BoundsInvariant(t *testing.T) {
	pool := mustCreatePool(t, "pool1", PoolMonolithic, 2, 5)

	// Start with 2 instances (at MinReplicas)
	inst1 := &InstanceSimulator{ID: "inst1"}
	inst2 := &InstanceSimulator{ID: "inst2"}
	if err := pool.AddInstance(inst1); err != nil {
		t.Fatalf("AddInstance(inst1) failed: %v", err)
	}
	if err := pool.AddInstance(inst2); err != nil {
		t.Fatalf("AddInstance(inst2) failed: %v", err)
	}

	// Verify invariant: MinReplicas <= Len() <= MaxReplicas
	if pool.Len() < pool.MinReplicas || pool.Len() > pool.MaxReplicas {
		t.Errorf("Invariant violated: pool size %d not in [%d, %d]", pool.Len(), pool.MinReplicas, pool.MaxReplicas)
	}

	// Add more instances
	inst3 := &InstanceSimulator{ID: "inst3"}
	inst4 := &InstanceSimulator{ID: "inst4"}
	if err := pool.AddInstance(inst3); err != nil {
		t.Fatalf("AddInstance(inst3) failed: %v", err)
	}
	if err := pool.AddInstance(inst4); err != nil {
		t.Fatalf("AddInstance(inst4) failed: %v", err)
	}

	if pool.Len() < pool.MinReplicas || pool.Len() > pool.MaxReplicas {
		t.Errorf("Invariant violated: pool size %d not in [%d, %d]", pool.Len(), pool.MinReplicas, pool.MaxReplicas)
	}

	// Remove instances
	if err := pool.RemoveInstance("inst4"); err != nil {
		t.Fatalf("RemoveInstance(inst4) failed: %v", err)
	}

	if pool.Len() < pool.MinReplicas || pool.Len() > pool.MaxReplicas {
		t.Errorf("Invariant violated: pool size %d not in [%d, %d]", pool.Len(), pool.MinReplicas, pool.MaxReplicas)
	}
}
