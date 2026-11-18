// Original to blis-sglang.

package hicache

import (
	"slices"
	"testing"
)

// Behavioural tests of the controller's rate limit, host release, storage backup and extra
// configuration (blis-sglang#109): each fails on a generated mutant of controller.go that the
// golden traces do not reach.

func newTestController(t *testing.T, page int) *Controller {
	t.Helper()
	tokens := page * 4
	host, err := NewHostPool(HostSizing{PageSize: page, HostTokenCapacity: &tokens})
	if err != nil {
		t.Fatal(err)
	}
	c, err := NewController(ControllerConfig{WritePolicy: "write_through", IOBackend: "direct", Layout: "layer_first",
		PageSize: page, TransferLayerIDMax: 2}, &Clock{}, host, host.Size(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestPrefetchIsRateLimitedFromTheMomentItFillsTheLimit(t *testing.T) {
	c := newTestController(t, 4)
	c.PrefetchCapacityLimit = 100
	for _, tc := range []struct {
		occupied int
		want     bool
	}{{0, false}, {99, false}, {100, true}, {101, true}} {
		c.PrefetchTokensOccupied = tc.occupied
		if got := c.PrefetchRateLimited(); got != tc.want {
			t.Errorf("occupied %d of limit 100: rate limited = %v, want %v", tc.occupied, got, tc.want)
		}
	}
}

func TestHostMemReleaseQueuesOnePagePerEntryAndNoEmptyOnes(t *testing.T) {
	c := newTestController(t, 2)
	c.AppendHostMemRelease([]int64{10, 11, 12, 13})
	want := [][]int64{{10, 11}, {12, 13}}
	if !slices.EqualFunc(c.HostMemReleaseQueue, want, slices.Equal[[]int64]) {
		t.Fatalf("two whole pages queued as %v, want %v", c.HostMemReleaseQueue, want)
	}
	c.HostMemReleaseQueue = nil
	c.AppendHostMemRelease(nil)
	if len(c.HostMemReleaseQueue) != 0 {
		t.Fatalf("no slots queued %v, want nothing", c.HostMemReleaseQueue)
	}
}

// batchLog is a storage backend that records each batch the backup writes.
type batchLog struct {
	Storage
	batches [][]string
	prefix  [][]string
}

func (s *batchLog) RegisterMemPoolHost(HostPoolPort)           {}
func (s *batchLog) RegisterMemHostPoolV2(HostPoolPort, string) {}
func (s *batchLog) OnHostPoolsRegistered()                     {}
func (s *batchLog) BatchSet(keys []string, _ []int64) bool {
	s.batches = append(s.batches, slices.Clone(keys))
	return true
}

func (s *batchLog) BatchSetV1(keys []string, _ []int64, extra *StorageExtraInfo) []bool {
	s.batches = append(s.batches, slices.Clone(keys))
	s.prefix = append(s.prefix, slices.Clone(extra.PrefixKeys))
	out := make([]bool, len(keys))
	for i := range out {
		out[i] = true
	}
	return out
}

func hashes(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = string(rune('A'+i%26)) + string(rune('a'+i/26%26)) + string(rune('0'+i/676))
	}
	return out
}

func backup(t *testing.T, backend string, n int, prefix []string) (*batchLog, *StorageOperation) {
	t.Helper()
	const page = 2
	c := newTestController(t, page)
	s := &batchLog{}
	if err := c.AttachStorage(backend, s, StorageConfig{}, 0); err != nil {
		t.Fatal(err)
	}
	c.WriteStorage(make([]int64, n*page), nil, hashes(n), prefix)
	c.BackupThreadFunc()
	if len(c.AckBackupQueue) != 1 {
		t.Fatalf("acked backups = %d, want 1", len(c.AckBackupQueue))
	}
	return s, c.AckBackupQueue[0]
}

func TestBackupWritesWholeBatchesAndNoEmptyOne(t *testing.T) {
	for _, backend := range []string{"file", "mooncake"} { // a page-data backend and a zero-copy one
		for _, n := range []int{0, StorageBatchSize, 2 * StorageBatchSize, 2*StorageBatchSize + 5} {
			s, op := backup(t, backend, n, nil)
			want := (n + StorageBatchSize - 1) / StorageBatchSize
			if len(s.batches) != want {
				t.Errorf("%s, %d pages: %d batches written, want %d", backend, n, len(s.batches), want)
			}
			for i, b := range s.batches {
				if len(b) == 0 {
					t.Errorf("%s, %d pages: batch %d is empty", backend, n, i)
				}
			}
			if op.CompletedTokens != 2*n {
				t.Errorf("%s, %d pages: completed tokens %d, want %d", backend, n, op.CompletedTokens, 2*n)
			}
		}
	}
}

func TestBackupGrowsThePrefixKeysOnlyWhenItWasGiven(t *testing.T) {
	n := 2*StorageBatchSize + 3
	all := hashes(n)

	s, _ := backup(t, "mooncake", n, []string{"root"})
	want := [][]string{{"root"}, append([]string{"root"}, all[:StorageBatchSize]...), append([]string{"root"}, all[:2*StorageBatchSize]...)}
	if !slices.EqualFunc(s.prefix, want, slices.Equal[[]string]) {
		t.Errorf("prefix keys per batch (lengths) = %v, want %v", lens(s.prefix), lens(want))
	}

	s, _ = backup(t, "mooncake", n, nil)
	for i, p := range s.prefix {
		if len(p) != 0 {
			t.Errorf("batch %d of a backup with no prefix keys was given %d prefix keys, want none", i, len(p))
		}
	}
}

func lens(x [][]string) []int {
	out := make([]int, len(x))
	for i := range x {
		out[i] = len(x[i])
	}
	return out
}

func TestStorageExtraConfigAcceptsNumbersWhereTheyBelong(t *testing.T) {
	c := newTestController(t, 4)
	got, err := c.ParseStorageBackendExtraConfig(`{"prefetch_threshold": 512, "prefetch_timeout_base": 3, "prefetch_timeout_per_ki_token": 0.5}`)
	if err != nil {
		t.Fatal(err)
	}
	if got.PrefetchThreshold != 512 || got.PrefetchTimeoutBase != 3 || got.PrefetchTimeoutPerKiToken != 0.5 {
		t.Fatalf("parsed %d, %v, %v; want 512, 3, 0.5", got.PrefetchThreshold, got.PrefetchTimeoutBase, got.PrefetchTimeoutPerKiToken)
	}
}

func TestStorageExtraConfigRefusesWhatIsNotTheRightNumber(t *testing.T) {
	c := newTestController(t, 4)
	for _, extra := range []string{
		`{"prefetch_threshold": "512"}`,
		`{"prefetch_threshold": 1.5}`,                   // not an int
		`{"prefetch_threshold": 100000000000000000000}`, // an int no int64 holds
		`{"prefetch_timeout_base": "1"}`,
		`{"prefetch_timeout_base": 1e999}`, // a number no float64 holds: refused, not stored as infinity
		`{"prefetch_timeout_per_ki_token": true}`,
		`{"prefetch_timeout_per_ki_token": -1e999}`,
	} {
		if got, err := c.ParseStorageBackendExtraConfig(extra); err == nil {
			t.Errorf("%s accepted as %+v", extra, got)
		}
	}
}
