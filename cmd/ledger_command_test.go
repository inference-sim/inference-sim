package cmd

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
	"testing/fstest"

	substrate "github.com/inference-sim/blis-substrate"
)

const (
	fixtureSeed = "5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed"
)

// checkout builds a substrate-shaped catalog directory in a temporary directory: the real
// ledger.yaml with a seed recorded, the real sources.yaml, the real unit list (linked), and the
// given published files (role -> file bytes). It returns the catalog directory.
func checkout(t *testing.T, published map[string][]byte) string {
	t.Helper()
	src := filepath.Join(substrateModuleDir(t), "catalog", "ledger")
	dir := t.TempDir()
	led := filepath.Join(dir, "ledger")
	if err := os.MkdirAll(filepath.Join(led, "published"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := readFile(t, filepath.Join(src, "ledger.yaml"))
	for _, k := range []string{"sha256", "generated_by", "generated_on", "rotation_owner"} {
		v := map[string]string{"sha256": fixtureSeed, "generated_by": "test", "generated_on": "2026-11-11", "rotation_owner": "test"}[k]
		re := regexp.MustCompile(`(?m)^(  ` + k + `:) ""`)
		if !re.MatchString(cfg) {
			t.Fatalf("ledger.yaml has no empty seed.%s to fill", k)
		}
		cfg = re.ReplaceAllString(cfg, `${1} "`+v+`"`)
	}
	for name, body := range map[string]string{"ledger.yaml": cfg, "sources.yaml": readFile(t, filepath.Join(src, "sources.yaml"))} {
		if err := os.WriteFile(filepath.Join(led, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// Eight units of the real list, so a check reads kilobytes, not the 82,331-line file.
	if err := os.MkdirAll(filepath.Join(led, "units"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(led, "units", "inferencex.jsonl.gz"), firstUnits(t, 8), 0o644); err != nil {
		t.Fatal(err)
	}
	for role, raw := range published {
		if err := os.WriteFile(filepath.Join(led, "published", role+".yaml"), raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// firstUnits is the real unit list cut to its first n lines, gzipped.
func firstUnits(t *testing.T, n int) []byte {
	t.Helper()
	f, err := os.Open(filepath.Join(substrateModuleDir(t), "catalog", "ledger", "units", "inferencex.jsonl.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	sc := bufio.NewScanner(zr)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for i := 0; i < n && sc.Scan(); i++ {
		_, _ = zw.Write(append(sc.Bytes(), '\n'))
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// sortedUnitIDs are the identifiers of the first n units of the real list, sorted: the units
// checkout() puts in its list.
func sortedUnitIDs(t *testing.T, n int) []string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "u.jsonl.gz")
	if err := os.WriteFile(path, firstUnits(t, 8), 0o644); err != nil {
		t.Fatal(err)
	}
	ids := readUnitIDs(t, path)
	sort.Strings(ids)
	return ids[:n]
}

// threeLists are valid published lists over real units: fit, anchor and report, disjoint, drawn
// under the fixture seed.
func threeLists(t *testing.T) map[string][]byte {
	ids := sortedUnitIDs(t, 6)
	list := func(role string, units ...string) []byte {
		var b strings.Builder
		b.WriteString("kind: PublishedUnits\nrole: " + role + "\npartition_version: 1\nseed_sha256: " + fixtureSeed + "\nunits:\n")
		for _, u := range units {
			b.WriteString("  - " + u + "\n")
		}
		return []byte(b.String())
	}
	return map[string][]byte{"fit": list("fit", ids[0], ids[1]), "anchor": list("anchor", ids[2], ids[3]), "report": list("report", ids[4], ids[5])}
}

// The command, given a substrate checkout, prints each list the catalog carries byte for byte.
// It runs the real cobra command, so the --catalog flag and RunE are part of what is tested.
func TestLedgerList_CommandPrintsThePublishedFileOfACheckout(t *testing.T) {
	lists := threeLists(t)
	dir := checkout(t, lists)
	for _, role := range []string{"fit", "anchor", "report"} {
		stdout, stderr, err := runRawCLI(t, "ledger", "--list", role, "--catalog", dir)
		if err != nil {
			t.Fatalf("%s: %v\n%s", role, err, stderr)
		}
		if stdout != string(lists[role]) {
			t.Errorf("%s: printed %q, want the file's bytes %q", role, stdout, lists[role])
		}
	}
}

// Known-bad catalogs, run through the command (X18): the refusals the substrate's
// LoadPublished makes are the command's. Nothing is printed beside a refusal.
func TestLedgerList_CommandRefusesWhatTheSubstrateRefuses(t *testing.T) {
	ids := sortedUnitIDs(t, 6)
	good := threeLists(t)
	with := func(role string, raw []byte) map[string][]byte {
		m := map[string][]byte{}
		for k, v := range good {
			m[k] = v
		}
		m[role] = raw
		return m
	}
	replaced := func(role, old, new string) map[string][]byte {
		raw := string(good[role])
		if !strings.Contains(raw, old) {
			t.Fatalf("%s list has no %q", role, old)
		}
		return with(role, []byte(strings.Replace(raw, old, new, 1)))
	}
	only := func(role string) map[string][]byte { return map[string][]byte{role: good[role]} }
	cases := map[string]struct {
		published map[string][]byte
		want      string
	}{
		"a seed that ledger.yaml does not record": {replaced("anchor", fixtureSeed, "deadbeef"), "is not the seed ledger.yaml records"},
		"a partition that ledger.yaml is not at":  {replaced("anchor", "partition_version: 1", "partition_version: 99"), "ledger.yaml is at 1"},
		"a unit that is in no source":             {replaced("anchor", ids[2], "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"), "in no source"},
		"one list of the three":                   {only("anchor"), "1 of the three"},
		"a unit on two lists":                     {replaced("report", ids[4], ids[2]), "has one role"},
		"a wrong kind":                            {replaced("anchor", "PublishedUnits", "Other"), "kind"},
		"the file of another role":                {with("anchor", good["fit"]), `role "fit"`},
		"a unit that is not an identifier":        {replaced("anchor", ids[3], "xyz"), "not a unit_id"},
		"units out of order":                      {with("anchor", []byte(strings.Replace(string(good["anchor"]), ids[2]+"\n  - "+ids[3], ids[3]+"\n  - "+ids[2], 1))), "out of order"},
		"an unknown key":                          {with("anchor", append(append([]byte(nil), good["anchor"]...), []byte("extra: 1\n")...)), "extra"},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			dir := checkout(t, c.published)
			var out bytes.Buffer
			err := printPublishedList(&out, dir, nil, "anchor")
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %v, want one containing %q", err, c.want)
			}
			if out.Len() != 0 {
				t.Errorf("printed %d bytes beside the refusal", out.Len())
			}
		})
	}
	// The case the substrate reader exists for: a directory holding only a list, with no
	// ledger.yaml to check its seed, partition and units against, is refused by the command.
	bare := t.TempDir()
	if err := os.MkdirAll(filepath.Join(bare, "ledger", "published"), 0o755); err != nil {
		t.Fatal(err)
	}
	bogus := "kind: PublishedUnits\nrole: anchor\npartition_version: 99\nseed_sha256: deadbeef\nunits:\n  - " + strings.Repeat("a", 64) + "\n"
	if err := os.WriteFile(filepath.Join(bare, "ledger", "published", "anchor.yaml"), []byte(bogus), 0o644); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := runRawCLI(t, "ledger", "--list", "anchor", "--catalog", bare)
	if err == nil || stdout != "" || !strings.Contains(stderr, "ledger.yaml") {
		t.Errorf("a list with a bogus seed and no ledger.yaml: err %v, stdout %q, stderr %q; want a refusal naming ledger.yaml", err, stdout, stderr)
	}
}

// A checkout that publishes no list, a role that is not published, and a request for no list at
// all are refused, with nothing printed.
func TestLedgerList_CommandRefusesAnAbsentOrUnpublishedList(t *testing.T) {
	dir := checkout(t, nil)
	for name, c := range map[string]struct {
		args []string
		want string
	}{
		"a checkout with no published lists": {[]string{"ledger", "--list", "anchor", "--catalog", dir}, "has no anchor list"},
		"a role that is not published":       {[]string{"ledger", "--list", "bogus", "--catalog", dir}, "fit, anchor and report"},
		"a role that is held, not published": {[]string{"ledger", "--list", "held", "--catalog", dir}, "fit, anchor and report"},
		"no list named":                      {[]string{"ledger"}, "--list"},
	} {
		stdout, stderr, err := runRawCLI(t, c.args...)
		if err == nil || stdout != "" || !strings.Contains(stderr, c.want) {
			t.Errorf("%s: err %v, stdout %q, stderr %q; want a refusal containing %q", name, err, stdout, stderr, c.want)
		}
	}
}

// The embedded copy has no ledger.yaml and no unit lists, so what it can be checked for is what
// the lists say of themselves and of each other; the substrate's LoadPublished does that.
func TestLedgerList_EmbeddedCopyIsCheckedForWhatItCanBe(t *testing.T) {
	good := threeLists(t)
	catalog := func(m map[string][]byte) fs.FS {
		f := fstest.MapFS{}
		for role, raw := range m {
			f["ledger/published/"+role+".yaml"] = &fstest.MapFile{Data: raw}
		}
		return f
	}
	var out bytes.Buffer
	if err := printPublishedList(&out, "", catalog(good), "anchor"); err != nil || out.String() != string(good["anchor"]) {
		t.Fatalf("three consistent embedded lists: err %v, printed %q", err, out.String())
	}
	other := strings.Replace(string(good["report"]), fixtureSeed, strings.Repeat("c", 64), 1)
	cases := map[string]struct {
		cat  fs.FS
		want string
	}{
		"a release that publishes no list": {fstest.MapFS{}, "has no anchor list"},
		"one list of three":                {catalog(map[string][]byte{"anchor": good["anchor"]}), "1 of the three"},
		"lists drawn under two seeds":      {catalog(map[string][]byte{"fit": good["fit"], "anchor": good["anchor"], "report": []byte(other)}), "seed"},
		"a wrong kind":                     {catalog(map[string][]byte{"fit": good["fit"], "anchor": []byte(strings.Replace(string(good["anchor"]), "PublishedUnits", "Other", 1)), "report": good["report"]}), "kind"},
	}
	for name, c := range cases {
		out.Reset()
		err := printPublishedList(&out, "", c.cat, "anchor")
		if err == nil || !strings.Contains(err.Error(), c.want) || out.Len() != 0 {
			t.Errorf("%s: err %v, printed %d bytes; want a refusal containing %q", name, err, out.Len(), c.want)
		}
	}
}

// The pinned substrate release is what the binary embeds. If it carries published lists, each
// is printed; if it carries none, `ledger --list` says so and fails. Either way the command
// prints nothing the release does not carry. The command runs without a catalog named.
func TestLedgerList_AgreesWithTheEmbeddedRelease(t *testing.T) {
	t.Setenv(catalogEnvVar, "")
	for _, role := range []string{"fit", "anchor", "report"} {
		raw, readErr := fs.ReadFile(substrate.Catalog(), "ledger/published/"+role+".yaml")
		stdout, _, err := runRawCLI(t, "ledger", "--list", role)
		switch {
		case readErr != nil && (err == nil || stdout != ""):
			t.Errorf("%s: the release carries no list and the command printed %q (err %v)", role, stdout, err)
		case readErr == nil && (err != nil || stdout != string(raw)):
			t.Errorf("%s: the release carries the list; the command printed %q, err %v", role, stdout, err)
		}
	}
}

// publishedFilesMissedBy returns the files under catalog/ledger/published/ that no pattern of
// patterns (path.Match syntax, relative to catalog/) selects. A published list the embedded
// patterns miss would be left out of the binary while the loader read it from a checkout.
func publishedFilesMissedBy(files, patterns []string) []string {
	var missed []string
	for _, f := range files {
		hit := false
		for _, p := range patterns {
			if ok, _ := path.Match(p, f); ok {
				hit = true
				break
			}
		}
		if !hit {
			missed = append(missed, f)
		}
	}
	sort.Strings(missed)
	return missed
}

// publishedFilesMissedBy is the guard behind SUB-9 for the lists: a file under
// catalog/ledger/published/ that the patterns the loader and the go:embed line share do not
// select is left out of the binary while a checkout loads it. The substrate's own test compares
// its go:embed directive with its patterns; this checks the patterns of the release pinned here
// against the published files of the same release, and has its known-bad input. At the pinned
// release no list is published, so the second half checks nothing yet (it logs 0 files).
func TestPublishedListsAreSelectedByTheReleasesPatterns(t *testing.T) {
	// Known-bad: the three files, against the patterns of a release that has not added them.
	files := []string{"ledger/published/anchor.yaml", "ledger/published/fit.yaml", "ledger/published/report.yaml"}
	if missed := publishedFilesMissedBy(files, []string{"devices/*.yaml", "models/*/graph.yaml"}); len(missed) != 3 {
		t.Fatalf("patterns that omit ledger/published/ missed %v; want all three", missed)
	}
	if missed := publishedFilesMissedBy(files, append([]string{"devices/*.yaml"}, "ledger/published/*.yaml")); len(missed) != 0 {
		t.Fatalf("patterns that include ledger/published/*.yaml missed %v", missed)
	}
	// The release pinned here: every published file it ships must be selected.
	dir := substrateModuleDir(t)
	matches, _ := filepath.Glob(filepath.Join(dir, "catalog", "ledger", "published", "*"))
	var shipped []string
	for _, m := range matches {
		shipped = append(shipped, "ledger/published/"+filepath.Base(m))
	}
	if missed := publishedFilesMissedBy(shipped, substrate.CatalogPatterns); len(missed) != 0 {
		t.Errorf("the pinned release ships %v under catalog/ledger/published/, which its patterns do not select", missed)
	}
	t.Logf("the pinned release ships %d published list file(s)", len(shipped))
}

func substrateModuleDir(t *testing.T) string {
	t.Helper()
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "github.com/inference-sim/blis-substrate").Output()
	if err != nil {
		t.Fatalf("go list -m: %v", err)
	}
	return strings.TrimSpace(string(out))
}

// readUnitIDs reads a unit list without the ledger package: gunzip, one JSON object a line, the
// unit_id of each.
func readUnitIDs(t *testing.T, file string) []string {
	t.Helper()
	f, err := os.Open(file)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	sc := bufio.NewScanner(zr)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	var ids []string
	for sc.Scan() {
		var u struct {
			UnitID string `json:"unit_id"`
		}
		if err := json.Unmarshal(sc.Bytes(), &u); err != nil {
			t.Fatal(err)
		}
		ids = append(ids, u.UnitID)
	}
	return ids
}

// `ledger units` prints every unit of the unit list with its identifier, sorted, and nothing
// else; it is checked against a reading of the file that shares no code with the command.
func TestLedgerUnits_PrintsEveryUnitWithItsIdentifier(t *testing.T) {
	cat := filepath.Join(substrateModuleDir(t), "catalog")
	var out bytes.Buffer
	if err := printUnits(&out, filepath.Join(cat, "ledger")); err != nil {
		t.Fatal(err)
	}
	want := readUnitIDs(t, filepath.Join(cat, "ledger", "units", "inferencex.jsonl.gz"))
	lines := strings.Split(strings.TrimSuffix(out.String(), "\n"), "\n")
	if len(want) < 1000 || len(lines) != len(want) {
		t.Fatalf("printed %d units, the file holds %d (non-vacuity: at least 1000)", len(lines), len(want))
	}
	for i, l := range lines {
		id, src, ok := strings.Cut(l, "\t")
		if !ok || src != "inferencex" || id != want[i] {
			t.Fatalf("line %d is %q, want %s\\tinferencex", i, l, want[i])
		}
	}
}

// Known-bad catalogs for `ledger units`: no sources file, a unit list that is not gzip, and a
// source with no unit list print nothing and fail.
func TestLedgerUnits_RefusesABrokenLedger(t *testing.T) {
	write := func(files map[string]string) string {
		dir := t.TempDir()
		for name, body := range files {
			p := filepath.Join(dir, name)
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		return dir
	}
	for name, dir := range map[string]string{
		"no sources.yaml": write(map[string]string{"README.md": "x"}),
		"a unit list that is not gzip": write(map[string]string{
			"sources.yaml": "kind: LedgerSources\nsources: []\n", "units/x.jsonl.gz": "not gzip"}),
	} {
		var out bytes.Buffer
		if err := printUnits(&out, dir); err == nil || out.Len() != 0 {
			t.Errorf("%s: err %v, printed %d bytes; want a refusal and no output", name, err, out.Len())
		}
	}
}

// With no catalog named, `ledger units` refuses: no release embeds the unit lists.
func TestLedgerUnits_NeedsACheckout(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestKernelCLILeg$")
	cmd.Env = append(os.Environ(), kernelCLIArgsEnv+"="+strings.Join([]string{"ledger", "units"}, "\x1f"), catalogEnvVar+"=")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err == nil || out.Len() != 0 {
		t.Errorf("ledger units with no catalog: err %v, output %d bytes; want a refusal and no output", err, out.Len())
	}
}

// goEmbedPatterns returns the patterns of the //go:embed lines of a Go source file.
func goEmbedPatterns(src string) []string {
	var out []string
	for _, line := range strings.Split(src, "\n") {
		if rest, ok := strings.CutPrefix(line, "//go:embed "); ok {
			out = append(out, strings.Fields(rest)...)
		}
	}
	sort.Strings(out)
	return out
}

// embedPatternsDiffer names the first difference between the go:embed patterns of a data.go and
// the loader's two pattern lists (the directory prefixed), or "" when they agree.
func embedPatternsDiffer(dataGo string, catalog, registry []string) string {
	var want []string
	for _, p := range catalog {
		want = append(want, "catalog/"+p)
	}
	for _, p := range registry {
		want = append(want, "registry/"+p)
	}
	sort.Strings(want)
	got := goEmbedPatterns(dataGo)
	in := map[string]bool{}
	for _, p := range got {
		in[p] = true
	}
	for _, p := range want {
		if !in[p] {
			return "the loader reads " + p + ", which the go:embed line does not embed"
		}
		delete(in, p)
	}
	for p := range in {
		return "the go:embed line embeds " + p + ", which the loader does not read"
	}
	return ""
}

// The patterns that decide what the pinned release embeds equal the patterns its loader reads
// (D7, SUB-9): a published list the loader reads from a checkout but the binary left out would
// make `ledger --list` print less offline than a checkout prints. The known-bad case is a data.go
// whose go:embed line leaves the published lists out of the loader's patterns.
func TestEmbedPatternsEqualTheLoadersPatterns(t *testing.T) {
	dataGo, err := os.ReadFile(filepath.Join(substrateModuleDir(t), "data.go"))
	if err != nil {
		t.Fatal(err)
	}
	if why := embedPatternsDiffer(string(dataGo), substrate.CatalogPatterns, substrate.RegistryPatterns); why != "" {
		t.Errorf("the pinned release's embed patterns differ from its loader's: %s", why)
	}
	bad := strings.Replace(string(dataGo), " catalog/workloads/*.yaml", "", 1)
	if why := embedPatternsDiffer(bad, substrate.CatalogPatterns, substrate.RegistryPatterns); !strings.Contains(why, "workloads") {
		t.Errorf("a go:embed line that omits a loader pattern was not found: %q", why)
	}
	extra := strings.Replace(string(dataGo), "//go:embed registry/", "//go:embed catalog/ledger/published/*.yaml registry/", 1)
	if why := embedPatternsDiffer(extra, substrate.CatalogPatterns, substrate.RegistryPatterns); !strings.Contains(why, "published") {
		t.Errorf("a go:embed line with a pattern the loader lacks was not found: %q", why)
	}
}
