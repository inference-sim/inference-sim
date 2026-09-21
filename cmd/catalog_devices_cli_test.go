package cmd

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// CLI-level contracts for #1770: a real `blis run` / `blis replay` resolves a
// --kv-offload-config `device_class` from the CATALOG (<catalog>/devices/storage.yaml).
//
//	BC-7  INV-6 — an offload run naming a device_class is unchanged by the cutover: two
//	      catalogs holding the same device table produce byte-identical stdout, and the
//	      numbers that reach the simulation are the catalog's (observable in the exported
//	      trace header, which is written from the resolved sim.KVOffloadConfig).
//	BC-8  INV-13 — run and replay share resolveKVOffloadConfig, so replaying a trace with
//	      the same --kv-offload-config succeeds: on replay the header is authoritative and
//	      a flag that resolves to different numbers is refused, so a PASS here can only
//	      mean replay resolved the class from the catalog to the run's numbers.
//	BC-9  a catalog whose table is missing is refused at the CLI naming the path (R1) —
//	      never a silent fall back to zero physics or to a built-in table.
//
// Why the trace header rather than a stdout diff for the "numbers reached the sim" half:
// the storage-device physics are not observable in the aggregate stdout of a small run
// (the secondary tier sees no traffic there), so a stdout-only comparison between two
// device classes would be vacuous. The header IS the resolved config, so asserting on it
// proves the catalog numbers were what the run was configured with — and the unit-level
// equivalence test (TestBundledCatalogDevices_MatchHistoricalDefaultsBlock) pins that
// those numbers are the pre-cutover ones.

// Environment variables driving the re-exec subprocess legs.
const (
	devicesCLILegEnv     = "BLIS_KVDEV_CLI_LEG"
	devicesCLICatalogEnv = "BLIS_KVDEV_CLI_CATALOG"
	devicesCLIOffloadEnv = "BLIS_KVDEV_CLI_OFFLOAD"
	devicesCLITraceEnv   = "BLIS_KVDEV_CLI_TRACE"
)

// devicesCLIModel is catalogued in the committed test catalog testdata/catalog/models/.
const devicesCLIModel = "qwen/qwen3-14b"

// writeOffloadConfigWithClass writes a --kv-offload-config file whose single fs tier
// resolves its physics from the named device_class, and returns the path.
func writeOffloadConfigWithClass(t *testing.T, deviceClass string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "offload.yaml")
	body := "kv_offload:\n" +
		"  cpu_bytes_to_use: 1073741824\n" +
		"  secondary_tiers:\n" +
		"    - type: fs\n" +
		"      root_dir: /mnt/kv\n" +
		"      direct_io: true\n" +
		"      device_class: " + deviceClass + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write offload config: %v", err)
	}
	return path
}

// newDeviceCatalog builds a catalog CLONE ROOT with (a) a models/ entry copied verbatim
// from the test catalog so the model resolves, and (b) a devices/storage.yaml holding the
// given table. Returns the root.
func newDeviceCatalog(t *testing.T, table string) string {
	t.Helper()
	root := writeCatalogStorageDevices(t, table)
	shortName := devicesCLIModel[strings.Index(devicesCLIModel, "/")+1:]
	src := filepath.Join("..", "testdata", "catalog", "models", shortName, hfConfigFile)
	content, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("read test catalog entry %s: %v", src, err)
	}
	entryDir := filepath.Join(root, catalogModelsSubdir, shortName)
	if err := os.MkdirAll(entryDir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", entryDir, err)
	}
	if err := os.WriteFile(filepath.Join(entryDir, hfConfigFile), content, 0o644); err != nil {
		t.Fatalf("write catalog entry: %v", err)
	}
	return root
}

// runDevicesCLILeg re-execs this test binary as the named leg, returning stdout, stderr
// and the run error (nil on success). Callers that require success assert on err
// themselves — the refusal legs need the failure.
func runDevicesCLILeg(t *testing.T, testName, leg, catalog, offloadPath, tracePrefix string) (stdout, stderr string, err error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^"+testName+"$")
	cmd.Env = append(os.Environ(),
		devicesCLILegEnv+"="+leg,
		devicesCLICatalogEnv+"="+catalog,
		devicesCLIOffloadEnv+"="+offloadPath,
		devicesCLITraceEnv+"="+tracePrefix,
		// Neutralize any ambient BLIS_CATALOG so a leg cannot pass via the environment
		// instead of the catalog it was handed.
		catalogEnvVar+"=",
	)
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	err = cmd.Run()
	return out.String(), errBuf.String(), err
}

// devicesCLISubprocess executes the leg named by the environment, if any, and reports
// whether it did (in which case the parent test body must not run). Legs terminate the
// process, so this never returns true.
func devicesCLISubprocess() bool {
	leg := os.Getenv(devicesCLILegEnv)
	if leg == "" {
		return false
	}
	catalog := os.Getenv(devicesCLICatalogEnv)
	offload := os.Getenv(devicesCLIOffloadEnv)
	tracePrefix := os.Getenv(devicesCLITraceEnv)

	base := []string{
		"--model", devicesCLIModel,
		"--hardware", "H100", "--tp", "1",
		"--seed", "42",
		"--defaults-filepath", "../defaults.yaml",
		"--catalog", catalog,
		"--kv-offload-config", offload,
	}
	var args []string
	switch leg {
	case "run", "run-export":
		args = append([]string{"run", "--num-requests", "20"}, base...)
		if leg == "run-export" {
			args = append(args, "--trace-output", tracePrefix)
		}
	case "replay":
		args = append([]string{"replay",
			"--trace-header", tracePrefix + ".yaml",
			"--trace-data", tracePrefix + ".csv"}, base...)
	default:
		os.Exit(2)
	}

	rootCmd.SetArgs(args)
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
	os.Exit(0)
	return true
}

// ---------------------------------------------------------------------------
// BC-7: INV-6 — the catalog table is what the run uses, and where the catalog
// sits is not an input to the simulation
// ---------------------------------------------------------------------------

// TestRunCmd_KVOffloadDeviceClass_ResolvesFromCatalog is the cutover guard at the
// `blis run` boundary. Three legs over one offload config naming device_class nvme_gen4:
//
//   - bundled catalog vs a temp clone-root catalog carrying the SAME table ⇒ byte-identical
//     stdout (INV-6: the catalog's location and layout are not simulation inputs);
//   - the exported trace header records 7000/5000/80 — the catalog's nvme_gen4 numbers,
//     i.e. exactly what the deleted defaults.yaml block supplied;
//   - a catalog whose nvme_gen4 carries DIFFERENT numbers records those instead, which is
//     the non-vacuity half: the table really is read per-run rather than baked in.
func TestRunCmd_KVOffloadDeviceClass_ResolvesFromCatalog(t *testing.T) {
	if devicesCLISubprocess() {
		return
	}
	const name = "TestRunCmd_KVOffloadDeviceClass_ResolvesFromCatalog"
	offload := writeOffloadConfigWithClass(t, "nvme_gen4")
	bundled := filepath.Join("..", "testdata", "catalog")

	// The temp catalog's table is a verbatim copy of the historical nvme_gen4 numbers.
	sameTable := newDeviceCatalog(t,
		"nvme_gen4: {read_bandwidth: 7.0e3, write_bandwidth: 5.0e3, base_latency: 80.0}\n")

	viaBundled, _, err := runDevicesCLILeg(t, name, "run", bundled, offload, "")
	if err != nil {
		t.Fatalf("bundled-catalog leg failed: %v\n%s", err, viaBundled)
	}
	viaTemp, tempErrOut, err := runDevicesCLILeg(t, name, "run", sameTable, offload, "")
	if err != nil {
		t.Fatalf("temp-catalog leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, viaTemp, tempErrOut)
	}

	// Non-vacuity: an empty or metric-less stdout would make the comparison trivial.
	if !strings.Contains(viaBundled, "completed_requests") {
		t.Fatalf("non-vacuity: offload run produced no metrics:\n%s", viaBundled)
	}
	if viaBundled != viaTemp {
		t.Errorf("stdout must be byte-identical for two catalogs holding the same device table (INV-6)\nbundled:\n%s\ntemp:\n%s",
			viaBundled, viaTemp)
	}

	// The resolved numbers reach the simulation: the exported header is written from the
	// resolved sim.KVOffloadConfig.
	tracePrefix := filepath.Join(t.TempDir(), "offload")
	if out, errOut, err := runDevicesCLILeg(t, name, "run-export", bundled, offload, tracePrefix); err != nil {
		t.Fatalf("export leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, out, errOut)
	}
	header := readTraceHeader(t, tracePrefix+".yaml")
	for _, frag := range []string{"device_class: nvme_gen4", "read_bandwidth: 7000", "write_bandwidth: 5000", "base_latency: 80"} {
		if !strings.Contains(header, frag) {
			t.Errorf("trace header must record the catalog's nvme_gen4 physics (%q missing):\n%s", frag, header)
		}
	}

	// Non-vacuity for the header assertion: a catalog with different numbers for the same
	// class produces a different header, so the values above are read, not hardcoded.
	otherTable := newDeviceCatalog(t,
		"nvme_gen4: {read_bandwidth: 1.25e3, write_bandwidth: 6.25e2, base_latency: 4242.0}\n")
	otherPrefix := filepath.Join(t.TempDir(), "offload-other")
	if out, errOut, err := runDevicesCLILeg(t, name, "run-export", otherTable, offload, otherPrefix); err != nil {
		t.Fatalf("other-table export leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, out, errOut)
	}
	otherHeader := readTraceHeader(t, otherPrefix+".yaml")
	for _, frag := range []string{"read_bandwidth: 1250", "write_bandwidth: 625", "base_latency: 4242"} {
		if !strings.Contains(otherHeader, frag) {
			t.Errorf("non-vacuity: a different catalog table must change the resolved physics (%q missing):\n%s", frag, otherHeader)
		}
	}
}

// readTraceHeader reads an exported TraceV2 header file as text. Kept as text because the
// assertions are about the recorded numbers being present, not about re-deriving the
// header's schema.
func readTraceHeader(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read trace header %s: %v", path, err)
	}
	return string(data)
}

// ---------------------------------------------------------------------------
// BC-8: INV-13 — run and replay resolve the same table
// ---------------------------------------------------------------------------

// TestReplayCmd_KVOffloadDeviceClass_MatchesRunResolution is the INV-13 half, and it is
// self-checking: on replay the trace header is authoritative and reconcileReplayKVOffload
// refuses a --kv-offload-config that resolves to anything other than the recorded config.
// So the success leg passes ONLY if replay resolved device_class nvme_gen4 from the
// catalog to byte-for-byte the numbers the exporting run resolved.
//
// The negative control makes that non-vacuous: replaying the same trace against a catalog
// whose nvme_gen4 carries different numbers is refused. If replay ignored the catalog (or
// both legs silently used zero physics), the control leg would pass and this test would
// fail — which is exactly the drift #1770 exists to prevent.
func TestReplayCmd_KVOffloadDeviceClass_MatchesRunResolution(t *testing.T) {
	if devicesCLISubprocess() {
		return
	}
	const name = "TestReplayCmd_KVOffloadDeviceClass_MatchesRunResolution"
	offload := writeOffloadConfigWithClass(t, "nvme_gen4")
	bundled := filepath.Join("..", "testdata", "catalog")

	tracePrefix := filepath.Join(t.TempDir(), "offload")
	if out, errOut, err := runDevicesCLILeg(t, name, "run-export", bundled, offload, tracePrefix); err != nil {
		t.Fatalf("export leg failed: %v\nstdout:\n%s\nstderr:\n%s", err, out, errOut)
	}
	if _, err := os.Stat(tracePrefix + ".csv"); err != nil {
		t.Fatalf("trace export produced no data file: %v", err)
	}

	// Same catalog table (a temp clone-root copy) ⇒ replay's flag reconciles with the header.
	sameTable := newDeviceCatalog(t,
		"nvme_gen4: {read_bandwidth: 7.0e3, write_bandwidth: 5.0e3, base_latency: 80.0}\n")
	out, errOut, err := runDevicesCLILeg(t, name, "replay", sameTable, offload, tracePrefix)
	if err != nil {
		t.Fatalf("replay with the same device table must reconcile with the header (INV-13): %v\nstdout:\n%s\nstderr:\n%s",
			err, out, errOut)
	}
	if !strings.Contains(out, "completed_requests") {
		t.Fatalf("non-vacuity: replay produced no metrics:\n%s", out)
	}

	// Different numbers for the same class ⇒ refused, naming the conflict.
	otherTable := newDeviceCatalog(t,
		"nvme_gen4: {read_bandwidth: 1.25e3, write_bandwidth: 6.25e2, base_latency: 4242.0}\n")
	_, controlErrOut, err := runDevicesCLILeg(t, name, "replay", otherTable, offload, tracePrefix)
	if err == nil {
		t.Error("non-vacuity: replay against a catalog whose device table differs from the " +
			"recorded config must be refused — a pass means replay is not reading the catalog")
	} else if !strings.Contains(controlErrOut, "kv-offload-config") {
		t.Errorf("refusal must name --kv-offload-config, got:\n%s", controlErrOut)
	}
}

// ---------------------------------------------------------------------------
// BC-9: a catalog with no device table is refused at the CLI
// ---------------------------------------------------------------------------

// TestRunCmd_KVOffloadDeviceClass_MissingCatalogTableIsRefused: pointing --catalog at a
// catalog that has models/ but no devices/storage.yaml, while naming a device_class, is a
// hard error naming the path. The alternative — resolving the class to a zero-valued
// device — is plausible-but-wrong storage physics with no diagnostic anywhere.
func TestRunCmd_KVOffloadDeviceClass_MissingCatalogTableIsRefused(t *testing.T) {
	if devicesCLISubprocess() {
		return
	}
	const name = "TestRunCmd_KVOffloadDeviceClass_MissingCatalogTableIsRefused"

	// A catalog with the model entry but no devices/ namespace.
	noDevices := newDeviceCatalog(t, "nvme_gen4: {read_bandwidth: 1.0, write_bandwidth: 1.0, base_latency: 1.0}\n")
	if err := os.RemoveAll(filepath.Join(noDevices, catalogDevicesSubdir)); err != nil {
		t.Fatalf("remove devices namespace: %v", err)
	}

	offload := writeOffloadConfigWithClass(t, "nvme_gen4")
	stdout, stderr, err := runDevicesCLILeg(t, name, "run", noDevices, offload, "")
	if err == nil {
		t.Fatalf("a device_class with no catalog device table must be refused:\n%s", stdout)
	}
	if want := catalogStorageDevicesPath(noDevices); !strings.Contains(stderr, want) {
		t.Errorf("refusal must name the path %q, got:\n%s", want, stderr)
	}

	// Negative control: restoring the table makes the SAME invocation succeed, so the
	// refusal is attributable to the missing table and not to anything else in the leg.
	restored := newDeviceCatalog(t, "nvme_gen4: {read_bandwidth: 7.0e3, write_bandwidth: 5.0e3, base_latency: 80.0}\n")
	if out, errOut, err := runDevicesCLILeg(t, name, "run", restored, offload, ""); err != nil {
		t.Errorf("negative control: the same run with a device table present must succeed: %v\nstdout:\n%s\nstderr:\n%s",
			err, out, errOut)
	}
}
