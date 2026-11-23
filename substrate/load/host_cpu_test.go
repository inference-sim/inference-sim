package load

import "testing"

// Every committed chip states the architecture of the CPU that drives it, because an engine-path
// coefficient is scoped by it: x86 for the HGX and DGX hosts of H100 and H200, Grace where the
// GPUs sit on a Grace CPU (GB200, GB300).
func TestEveryCatalogChipStatesItsHostCPU(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"a100-80": "x86", "a100-sxm": "x86", "h100": "x86", "h200": "x86", "l40s": "x86",
		"rtx6000pro-server": "x86", "b200": "x86", "b300": "x86",
		"gb200-nvl72": "grace", "gb300": "grace",
	}
	for name, cpu := range want {
		chip, err := d.Chip(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if chip.HostCPU != cpu {
			t.Errorf("%s: HostCPU = %q, want %q", name, chip.HostCPU, cpu)
		}
	}
	for _, name := range d.Chips() {
		if _, ok := want[name]; !ok {
			t.Errorf("chip %s is in the catalog but this test does not state its host CPU", name)
		}
	}
}
