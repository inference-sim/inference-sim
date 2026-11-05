package workload

import (
	"reflect"
	"sort"
	"testing"
)

// TestSIM11_EveryTraceFieldHasAStatus: every field of every TraceV2 struct replay reads has a
// status (consumed, refused or inert) with a reason, and every name in a table is a field (SIM-11).
// A field added to a trace struct without a status fails here.
func TestSIM11_EveryTraceFieldHasAStatus(t *testing.T) {
	for _, typ := range []reflect.Type{
		reflect.TypeOf(TraceHeader{}), reflect.TypeOf(TraceRouterConfig{}), reflect.TypeOf(TraceKVOffloadConfig{}),
		reflect.TypeOf(TraceKVOffloadTier{}), reflect.TypeOf(TraceRecord{}),
	} {
		table, ok := TraceFieldStatus(typ.Name())
		if !ok {
			t.Errorf("%s has no field-status table", typ.Name())
			continue
		}
		fields := map[string]bool{}
		for i := 0; i < typ.NumField(); i++ {
			name := typ.Field(i).Name
			fields[name] = true
			e, ok := table[name]
			switch {
			case !ok:
				t.Errorf("%s.%s has no status: add it to the table in fieldstatus.go as consumed, refused or inert", typ.Name(), name)
			case e.Status != FieldConsumed && e.Status != FieldRefused && e.Status != FieldInert:
				t.Errorf("%s.%s: status %q is not consumed, refused or inert", typ.Name(), name, e.Status)
			case e.Why == "":
				t.Errorf("%s.%s: the status states no reason", typ.Name(), name)
			}
		}
		names := make([]string, 0, len(table))
		for name := range table {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			if !fields[name] {
				t.Errorf("%s's table names %s, which is not a field", typ.Name(), name)
			}
		}
	}
}

// TestTraceFieldStatus_ReturnsACopy: the tables are not exported mutable maps (R8).
func TestTraceFieldStatus_ReturnsACopy(t *testing.T) {
	m, _ := TraceFieldStatus("TraceRecord")
	m["InputTokens"] = FieldEntry{FieldInert, "changed"}
	again, _ := TraceFieldStatus("TraceRecord")
	if again["InputTokens"].Status != FieldConsumed {
		t.Error("a caller's change to the returned table reached the package's table")
	}
}
