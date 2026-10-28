package cmd

import (
	"reflect"
	"testing"
)

// BC-D6/BC-G6: the resolved device-model fields (non-default Qsat/f₁/σ) round-trip
// losslessly through the trace-header converters, for both regimes.
func TestKVOffloadHeaderConversion_DeviceModelRoundTrip(t *testing.T) {
	for _, directIO := range []bool{true, false} {
		cfg, err := resolveKVOffload(rampBlock("nvme_ramp", directIO), deviceModelDevices(), 16)
		if err != nil {
			t.Fatal(err)
		}
		back := headerToSimOffload(simToHeaderOffload(cfg))
		if !reflect.DeepEqual(cfg, back) {
			t.Errorf("directIO=%v: device-model round-trip not identity:\n got  %+v\n want %+v", directIO, back.Tiers[0], cfg.Tiers[0])
		}
	}
}
