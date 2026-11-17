package cmd

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// CLI-level contracts for #1770: a real `blis run` resolves the storage devices its scenario's
// offload block names from the CATALOG (<catalog>/devices/storage.yaml).
//
//	BC-9  a catalog whose table is missing or malformed is refused at the CLI naming the path
//	      (R1) — never a silent fall back to zero physics or to a built-in table — and a tier
//	      naming a device the table does not define is refused naming it.
//
// The scenario is the only source of the offload tiers (#2037). The v0.33.0 engine models the
// cpu_dram tier only and refuses a secondary tier (blis-vllm#63), so these legs offload to
// cpu_dram, except where a refusal of the named device comes first.

// Environment variables driving the re-exec subprocess legs.
const (
	devicesCLILegEnv       = "BLIS_KVDEV_CLI_LEG"
	devicesCLICatalogEnv   = "BLIS_KVDEV_CLI_CATALOG"
	devicesCLIScenariosEnv = "BLIS_KVDEV_CLI_SCENARIOS"
)

// cpuTierOffload is an offload block with the cpu_dram tier alone.
const cpuTierOffload = "offload:\n  connector: OffloadingConnector\n  spec: CPUOffloadingSpec\n  tiers:\n    - tier: cpu_dram\n      bytes: 1073741824\n"

// offloadScenarios writes the test scenario with offload, an offload block, into a fresh
// scenario directory and returns it.
func offloadScenarios(t *testing.T, offload string) string {
	t.Helper()
	return writeOffloadScenario(t, testScenariosDir(t), kernelTestScenario, "", offload)
}

// newDeviceCatalog builds a catalog CLONE ROOT that is the vendored catalog -- so the kernel
// scenario's model graph, hardware and fabric resolve -- with devices/storage.yaml replaced by
// the given table plus the vendored cpu_dram entry, which every offload run loads its CPU-tier
// hits over (INV-K7). Returns the root.
func newDeviceCatalog(t *testing.T, table string) string {
	t.Helper()
	root := copyKernelCatalog(t)
	table += "cpu_dram: {_comment: \"a test class\"}\n"
	if err := os.WriteFile(catalogStorageDevicesPath(root), []byte(table), 0o644); err != nil {
		t.Fatalf("write storage.yaml: %v", err)
	}
	return root
}

// runDevicesCLILeg re-execs this test binary as a `blis run` over the scenarios directory
// and the catalog, returning stdout, stderr and the run error (nil on success).
func runDevicesCLILeg(t *testing.T, testName, catalog, scenarios string) (stdout, stderr string, err error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^"+testName+"$")
	cmd.Env = append(os.Environ(),
		devicesCLILegEnv+"=run",
		devicesCLICatalogEnv+"="+catalog,
		devicesCLIScenariosEnv+"="+scenarios,
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
	if os.Getenv(devicesCLILegEnv) == "" {
		return false
	}
	rootCmd.SetArgs([]string{"run", "--num-requests", "20",
		"--scenario", kernelTestScenario, "--scenarios", os.Getenv(devicesCLIScenariosEnv),
		"--registry", testRegistryDir(), "--seed", "42", "--defaults-filepath", "../defaults.yaml",
		"--catalog", os.Getenv(devicesCLICatalogEnv)})
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
	os.Exit(0)
	return true
}

// TestRunCmd_KVOffloadDeviceClass_MissingCatalogTableIsRefused: pointing --catalog at a
// catalog that has models/ but no devices/storage.yaml, for a scenario that offloads, is a
// hard error naming the path. The alternative — resolving the tier to a zero-valued device —
// is plausible-but-wrong storage physics with no diagnostic anywhere.
func TestRunCmd_KVOffloadDeviceClass_MissingCatalogTableIsRefused(t *testing.T) {
	if devicesCLISubprocess() {
		return
	}
	const name = "TestRunCmd_KVOffloadDeviceClass_MissingCatalogTableIsRefused"

	// A catalog with the model entry but no devices/ namespace.
	noDevices := newDeviceCatalog(t, "nvme_gen4: {_comment: \"a test class\"}\n")
	if err := os.RemoveAll(filepath.Join(noDevices, catalogDevicesSubdir)); err != nil {
		t.Fatalf("remove devices namespace: %v", err)
	}

	scenarios := offloadScenarios(t, cpuTierOffload)
	stdout, stderr, err := runDevicesCLILeg(t, name, noDevices, scenarios)
	if err == nil {
		t.Fatalf("an offloading run with no catalog device table must be refused:\n%s", stdout)
	}
	// blis-substrate's loader opens the catalog whole when the run starts and names each
	// file relative to the catalog it was refused in.
	for _, want := range []string{filepath.Base(noDevices), catalogStorageDevicesRelPath} {
		if !strings.Contains(stderr, want) {
			t.Errorf("refusal must name %q, got:\n%s", want, stderr)
		}
	}

	// Negative control: restoring the table makes the SAME invocation succeed, so the
	// refusal is attributable to the missing table and not to anything else in the leg.
	restored := newDeviceCatalog(t, "nvme_gen4: {_comment: \"a test class\"}\n")
	if out, errOut, err := runDevicesCLILeg(t, name, restored, scenarios); err != nil {
		t.Errorf("negative control: the same run with a device table present must succeed: %v\nstdout:\n%s\nstderr:\n%s",
			err, out, errOut)
	}
}

// TestRunCmd_KVOffloadRefusalsReachTheCLI: the KV offload configuration is resolved by the
// engine (blis-vllm compat/build, called from cmd/engine_build.go), and each kind of refusal
// ends `blis run` with it on stderr, naming what to fix: a malformed storage-device table (the
// loader opens the catalog first, so its decoding error names the catalog and the file in it),
// and a tier naming a device class the table does not define. A refusal is never a silent
// fallback.
func TestRunCmd_KVOffloadRefusalsReachTheCLI(t *testing.T) {
	if devicesCLISubprocess() {
		return
	}
	const name = "TestRunCmd_KVOffloadRefusalsReachTheCLI"
	good := "nvme_gen4: {_comment: \"a test class\"}\n"
	malformed := newDeviceCatalog(t, good)
	if err := os.WriteFile(catalogStorageDevicesPath(malformed), []byte("nvme_gen4: {_comment: [\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	floppy := strings.Replace(cpuTierOffload, "spec: CPUOffloadingSpec", "spec: TieringOffloadingSpec", 1) +
		"    - tier: floppy\n      bytes: 1073741824\n"
	for _, c := range []struct {
		name, catalog, offload string
		want                   []string
	}{
		{"a malformed device table", malformed, cpuTierOffload,
			[]string{malformed + " is refused", catalogStorageDevicesRelPath + ": yaml:", "did not find expected node content"}},
		{"an unknown device class", newDeviceCatalog(t, good), floppy, []string{"floppy"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			stdout, stderr, err := runDevicesCLILeg(t, name, c.catalog, offloadScenarios(t, c.offload))
			if err == nil {
				t.Fatalf("the run must be refused:\n%s", stdout)
			}
			for _, w := range c.want {
				if !strings.Contains(stderr, w) {
					t.Errorf("stderr must contain %q:\n%s", w, lastLines(stderr, 3))
				}
			}
		})
	}
}
