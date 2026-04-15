package cluster

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCacheEventArrivalEvent_TimestampAndPriority(t *testing.T) {
	event := &CacheEventArrivalEvent{
		time:       100_000,
		instanceID: "inst-1",
	}
	assert.Equal(t, int64(100_000), event.Timestamp())
	assert.Equal(t, PriorityCacheEvent, event.Priority())
	assert.Equal(t, 10, event.Priority(), "cache event priority should be 10 (after all existing events)")
}

func TestCacheEventArrivalEvent_Execute_NilStaleCache(t *testing.T) {
	// GIVEN a ClusterSimulator with nil staleCache (oracle mode)
	cs := &ClusterSimulator{}

	event := &CacheEventArrivalEvent{
		time:       100_000,
		instanceID: "inst-1",
	}

	// WHEN Execute is called
	// THEN it should not panic (defensive no-op)
	event.Execute(cs)
}
