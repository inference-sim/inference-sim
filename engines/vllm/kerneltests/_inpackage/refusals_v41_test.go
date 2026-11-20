// Moved from blis-latency-kernel/refusals_v41_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

// A term the kernel cannot price from its inputs is refused or recorded, never charged at
// nothing in silence.

// recorded reports whether a kernel's provenance names an entry.
func recorded(k *Kernel, name string) bool {
	for _, o := range k.Provenance() {
		if o.Name == name {
			return true
		}
	}
	return false
}
