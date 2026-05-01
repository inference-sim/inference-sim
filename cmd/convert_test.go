package cmd

import "testing"

// TestConvertServeGenCmd_NewFlagsRegistered verifies BC-12: new flags
// --window-duration-seconds and --drain-timeout-seconds are registered
// with correct defaults.
func TestConvertServeGenCmd_NewFlagsRegistered(t *testing.T) {
	windowFlag := convertServeGenCmd.Flags().Lookup("window-duration-seconds")
	if windowFlag == nil {
		t.Fatal("flag --window-duration-seconds not found")
	}
	if windowFlag.DefValue != "600" {
		t.Errorf("--window-duration-seconds default: got %q, want \"600\"", windowFlag.DefValue)
	}

	drainFlag := convertServeGenCmd.Flags().Lookup("drain-timeout-seconds")
	if drainFlag == nil {
		t.Fatal("flag --drain-timeout-seconds not found")
	}
	if drainFlag.DefValue != "180" {
		t.Errorf("--drain-timeout-seconds default: got %q, want \"180\"", drainFlag.DefValue)
	}
}

// TestConvertServeGenCmd_TimeFlagRemoved verifies BC-12: old --time flag is removed.
func TestConvertServeGenCmd_TimeFlagRemoved(t *testing.T) {
	if convertServeGenCmd.Flags().Lookup("time") != nil {
		t.Error("--time flag should be removed but still exists")
	}
}
