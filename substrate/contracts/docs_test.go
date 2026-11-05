package contracts

// The documentation under docs/ makes two kinds of claim this file checks. Each example
// document must load and validate without a single finding, so a reader who copies one
// starts from something correct. And each field table must list exactly the YAML keys its
// Go type declares — no key missing, none invented — so a field added to a type cannot
// ship undocumented and a renamed one cannot linger on the site.

import (
	"bufio"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/connector"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/evaluation"
	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/pointrecord"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/spec/router"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/trace"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// fixture returns a loaded document, panicking on a load error: every path here is committed,
// so a failure is a broken checkout rather than a finding.
func fixture[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func requireSilent(t *testing.T, name string, rep Report) {
	t.Helper()
	for _, p := range rep.Problems() {
		t.Errorf("%s: %s", name, p)
	}
}

func TestDocExamplesValidate(t *testing.T) {
	ex := func(name string) string { return filepath.Join("..", "docs", "contracts", "examples", name) }

	t.Run("colocated", func(t *testing.T) {
		requireSilent(t, "colocated", Validate(Bundle{Rules: openAnyEngine,
			Scenario:   fixture(LoadScenario(ex("scenario.yaml"))),
			Deployment: fixture(LoadDeployment(ex("deployment.yaml"))),
			Model:      fixture(LoadModelGraph("../catalog/models/llama-3.1-8b-instruct/graph.yaml")),
			Chip:       fixture(LoadChip("../catalog/hardware/h100.yaml")),
		}))
	})
	t.Run("disaggregated", func(t *testing.T) {
		requireSilent(t, "disaggregated", Validate(Bundle{Rules: openAnyEngine,
			Scenario:   fixture(LoadScenario(ex("scenario-disaggregated.yaml"))),
			Deployment: fixture(LoadDeployment(ex("deployment-disaggregated.yaml"))),
			Model:      fixture(LoadModelGraph("../catalog/models/deepseek-v3/graph.yaml")),
			Chip:       fixture(LoadChip("../catalog/hardware/h200.yaml")),
			Fabric:     fixture(LoadFabric("../catalog/networks/ib-400g.yaml")),
			Devices:    fixture(LoadStorageDevices("../catalog/devices/storage.yaml")),
		}))
	})
	t.Run("coefficient set", func(t *testing.T) {
		requireSilent(t, "coefficient set", Validate(Bundle{
			Coefficients: []*coefficient.Set{fixture(LoadCoefficientSet(ex("coefficient-set.yaml")))},
		}))
	})
	t.Run("acceptance profile set", func(t *testing.T) {
		requireSilent(t, "acceptance profile set", Validate(Bundle{
			AcceptanceProfiles: []*acceptance.ProfileSet{fixture(LoadAcceptanceProfileSet(ex("acceptance-profile-set.yaml")))},
		}))
	})
	t.Run("replay", func(t *testing.T) {
		requireSilent(t, "replay", Validate(Bundle{Rules: openAnyEngine,
			Scenario: fixture(LoadScenario(ex("replay-binding.yaml"))),
		}))
	})
	t.Run("catalog replay and a reference to it", func(t *testing.T) {
		w := fixture(LoadCatalogWorkload(ex("workload-replay.yaml")))
		requireSilent(t, "catalog replay", Validate(Bundle{ReplayWorkload: w.Replay}))
		s := fixture(LoadScenario(ex("replay-reference.yaml")))
		requireSilent(t, "replay reference", Validate(Bundle{Scenario: s, Rules: openAnyEngine}))
		root := t.TempDir()
		if err := os.MkdirAll(filepath.Join(root, "workloads"), 0o755); err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(ex("workload-replay.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "workloads", "workload-replay.yaml"), b, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ResolveReplay(root, s.Workload.Replay); err != nil {
			t.Errorf("the reference example does not resolve against the catalog example: %v", err)
		}
	})
	t.Run("acceptance pin", func(t *testing.T) {
		requireSilent(t, "acceptance pin", Validate(Bundle{Rules: openAnyEngine,
			Scenario: fixture(LoadScenario(ex("scenario-acceptance.yaml"))),
		}))
	})
	t.Run("evaluation run", func(t *testing.T) {
		requireSilent(t, "evaluation run", Validate(Bundle{
			Evaluation: fixture(LoadEvaluationRun(ex("evaluation-run.yaml"))),
		}))
	})
	t.Run("measured run and its licence registry", func(t *testing.T) {
		reg := fixture(LoadMeasurementSources(ex("measurement-sources.yaml")))
		for _, p := range reg.Validate().All() {
			t.Errorf("measurement-sources.yaml: %s", p)
		}
		run := fixture(LoadMeasuredRun(ex("measured-run.yaml")))
		for _, p := range run.ValidateWithSources(reg).All() {
			t.Errorf("measured-run.yaml: %s", p)
		}
	})
	t.Run("model graph", func(t *testing.T) {
		requireSilent(t, "model graph", Validate(Bundle{
			Model: fixture(LoadModelGraph(ex("model-graph-deepseek-v4.1.yaml"))),
		}))
	})
	t.Run("connector registry", func(t *testing.T) {
		r := fixture(LoadConnectorRegistry(ex("connector-registry.yaml")))
		for _, p := range r.Validate().All() {
			t.Errorf("connector-registry.yaml: %s", p)
		}
	})
	t.Run("result", func(t *testing.T) {
		requireSilent(t, "result", Validate(Bundle{
			Result: fixture(LoadResult(ex("result.json"))),
		}))
	})
	t.Run("field list", func(t *testing.T) {
		if _, err := fields.Load(ex("field-list.json")); err != nil {
			t.Error(err)
		}
	})
	t.Run("trace", func(t *testing.T) {
		requireSilent(t, "trace", Validate(Bundle{Rules: openAnyEngine,
			Scenario: fixture(LoadScenario(ex("trace-binding.yaml"))),
		}))
	})
	t.Run("workload specification", func(t *testing.T) {
		s := fixture(LoadWorkloadSpec(ex("workload-spec.yaml")))
		for _, p := range s.Validate().All() {
			t.Errorf("workload-spec.yaml: %s", p)
		}
	})
	t.Run("point record", func(t *testing.T) {
		r := fixture(LoadPointRecord(ex("point-record.json")))
		for _, p := range r.Validate().All() {
			t.Errorf("point-record.json: %s", p)
		}
	})
	t.Run("trace file", func(t *testing.T) {
		h, err := os.Open(ex("trace-header.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		defer h.Close()
		d, err := os.Open(ex("trace-data.csv"))
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		if _, err := trace.Read(h, d); err != nil {
			t.Error(err)
		}
	})
	// Every file in docs/examples/ must be one of the cases above, so an example added
	// later cannot be shown on a page without ever being validated.
	covered := map[string]bool{
		"scenario.yaml": true, "deployment.yaml": true,
		"scenario-disaggregated.yaml": true, "deployment-disaggregated.yaml": true,
		"coefficient-set.yaml": true, "evaluation-run.yaml": true, "trace-binding.yaml": true,
		"model-graph-deepseek-v4.1.yaml": true,
		"acceptance-profile-set.yaml":    true, "scenario-acceptance.yaml": true,
		"replay-binding.yaml": true, "workload-replay.yaml": true, "replay-reference.yaml": true,
		"measured-run.yaml": true, "measurement-sources.yaml": true,
		"connector-registry.yaml": true,
		"result.json":             true,
		"field-list.json":         true,
		"trace-header.yaml":       true, "trace-data.csv": true, "workload-spec.yaml": true, "point-record.json": true,
	}
	all, err := filepath.Glob(ex("*"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range all {
		if !covered[filepath.Base(path)] {
			t.Errorf("%s is not validated by TestDocExamplesValidate; add a case for it", path)
		}
	}

	// The by-name references the examples make must resolve against the vendored catalog,
	// or an example would name a chip or a workload the reader cannot find.
	for _, name := range []string{"scenario.yaml", "scenario-disaggregated.yaml", "trace-binding.yaml",
		"scenario-acceptance.yaml", "replay-binding.yaml"} {
		s := fixture(LoadScenario(ex(name)))
		rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, "")
		for _, p := range rep.Problems() {
			if strings.HasPrefix(p.Path, "references.coefficients") {
				continue // the registry is not vendored
			}
			t.Errorf("%s: %s", name, p)
		}
	}
}

// documented maps each field-table marker to the type whose keys the table must list, and
// the keys deliberately left out of it. A key is left out only when the file never carries
// it: an entry's name is its map key, and a storage tier's name is its key in the file.
var documented = map[string]struct {
	typ  reflect.Type
	omit []string
}{
	"scenario.Scenario":  {typ: reflect.TypeOf(scenario.Scenario{})},
	"scenario.Cluster":   {typ: reflect.TypeOf(scenario.Cluster{})},
	"scenario.Packaging": {typ: reflect.TypeOf(scenario.Packaging{})},
	"scenario.Engine":    {typ: reflect.TypeOf(scenario.Engine{})},
	"workload.Binding":   {typ: reflect.TypeOf(workload.Binding{})},
	"workload.TraceRef":  {typ: reflect.TypeOf(workload.TraceRef{})},
	"workload.Replay": {typ: reflect.TypeOf(workload.Replay{}),
		omit: keyList(reflect.TypeOf(workload.ReplayDefinition{}))},
	"workload.ReplayDefinition":   {typ: reflect.TypeOf(workload.ReplayDefinition{})},
	"workload.Corpus":             {typ: reflect.TypeOf(workload.Corpus{})},
	"workload.CorpusSource":       {typ: reflect.TypeOf(workload.CorpusSource{})},
	"workload.TrajectoryStart":    {typ: reflect.TypeOf(workload.TrajectoryStart{})},
	"workload.Warmup":             {typ: reflect.TypeOf(workload.Warmup{})},
	"trace.Header":                {typ: reflect.TypeOf(trace.Header{})},
	"trace.KVOffload":             {typ: reflect.TypeOf(trace.KVOffload{})},
	"trace.KVOffloadTier":         {typ: reflect.TypeOf(trace.KVOffloadTier{})},
	"trace.Router":                {typ: reflect.TypeOf(trace.Router{})},
	"trace.Record":                {typ: reflect.TypeOf(trace.Record{})},
	"workload.TokenBlocks":        {typ: reflect.TypeOf(workload.TokenBlocks{})},
	"trace.SLODimTargets":         {typ: reflect.TypeOf(trace.SLODimTargets{})},
	"workload.Spec":               {typ: reflect.TypeOf(workload.Spec{})},
	"workload.ClientSpec":         {typ: reflect.TypeOf(workload.ClientSpec{})},
	"workload.CohortSpec":         {typ: reflect.TypeOf(workload.CohortSpec{})},
	"workload.DiurnalSpec":        {typ: reflect.TypeOf(workload.DiurnalSpec{})},
	"workload.SpikeSpec":          {typ: reflect.TypeOf(workload.SpikeSpec{})},
	"workload.DrainSpec":          {typ: reflect.TypeOf(workload.DrainSpec{})},
	"workload.ArrivalSpec":        {typ: reflect.TypeOf(workload.ArrivalSpec{})},
	"workload.DistSpec":           {typ: reflect.TypeOf(workload.DistSpec{})},
	"workload.NetworkSpec":        {typ: reflect.TypeOf(workload.NetworkSpec{})},
	"workload.LifecycleSpec":      {typ: reflect.TypeOf(workload.LifecycleSpec{})},
	"workload.ActiveWindow":       {typ: reflect.TypeOf(workload.ActiveWindow{})},
	"workload.MultimodalSpec":     {typ: reflect.TypeOf(workload.MultimodalSpec{})},
	"workload.ReasoningSpec":      {typ: reflect.TypeOf(workload.ReasoningSpec{})},
	"workload.MultiTurnSpec":      {typ: reflect.TypeOf(workload.MultiTurnSpec{})},
	"workload.ServeGenDataSpec":   {typ: reflect.TypeOf(workload.ServeGenDataSpec{})},
	"workload.InferencePerfSpec":  {typ: reflect.TypeOf(workload.InferencePerfSpec{})},
	"workload.StageSpec":          {typ: reflect.TypeOf(workload.StageSpec{})},
	"workload.SharedPrefixSpec":   {typ: reflect.TypeOf(workload.SharedPrefixSpec{})},
	"workload.Shape":              {typ: reflect.TypeOf(workload.Shape{})},
	"workload.Distribution":       {typ: reflect.TypeOf(workload.Distribution{})},
	"deployment.Deployment":       {typ: reflect.TypeOf(deployment.Deployment{})},
	"deployment.Pool":             {typ: reflect.TypeOf(deployment.Pool{})},
	"deployment.Parallelism":      {typ: reflect.TypeOf(deployment.Parallelism{})},
	"deployment.Offload":          {typ: reflect.TypeOf(deployment.Offload{})},
	"deployment.Tier":             {typ: reflect.TypeOf(deployment.Tier{})},
	"deployment.PDTransfer":       {typ: reflect.TypeOf(deployment.PDTransfer{})},
	"deployment.Router":           {typ: reflect.TypeOf(deployment.Router{})},
	"model.Graph":                 {typ: reflect.TypeOf(model.Graph{})},
	"model.Licence":               {typ: reflect.TypeOf(model.Licence{})},
	"model.ConfigPin":             {typ: reflect.TypeOf(model.ConfigPin{})},
	"model.Derivation":            {typ: reflect.TypeOf(model.Derivation{})},
	"model.GlobalShape":           {typ: reflect.TypeOf(model.GlobalShape{})},
	"model.LayerKind":             {typ: reflect.TypeOf(model.LayerKind{})},
	"model.Stack":                 {typ: reflect.TypeOf(model.Stack{})},
	"model.Speculator":            {typ: reflect.TypeOf(model.Speculator{})},
	"model.Node":                  {typ: reflect.TypeOf(model.Node{})},
	"model.Cache":                 {typ: reflect.TypeOf(model.Cache{})},
	"model.ContextWork":           {typ: reflect.TypeOf(model.ContextWork{})},
	"model.WeightScale":           {typ: reflect.TypeOf(model.WeightScale{})},
	"model.ShardedAtDecode":       {typ: reflect.TypeOf(model.ShardedAtDecode{})},
	"model.ArchOverride":          {typ: reflect.TypeOf(model.ArchOverride{})},
	"model.NodeOverride":          {typ: reflect.TypeOf(model.NodeOverride{})},
	"model.CacheOverride":         {typ: reflect.TypeOf(model.CacheOverride{})},
	"model.Identity":              {typ: reflect.TypeOf(model.Identity{})},
	"model.Source":                {typ: reflect.TypeOf(model.Source{})},
	"hardware.Chip":               {typ: reflect.TypeOf(hardware.Chip{})},
	"hardware.Fabric":             {typ: reflect.TypeOf(hardware.Fabric{})},
	"hardware.StorageDevice":      {typ: reflect.TypeOf(hardware.StorageDevice{}), omit: []string{"name"}},
	"coefficient.Set":             {typ: reflect.TypeOf(coefficient.Set{})},
	"coefficient.Entry":           {typ: reflect.TypeOf(coefficient.Entry{}), omit: []string{"name"}},
	"coefficient.Source":          {typ: reflect.TypeOf(coefficient.Source{})},
	"coefficient.Scope":           {typ: reflect.TypeOf(coefficient.Scope{})},
	"evaluation.Run":              {typ: reflect.TypeOf(evaluation.Run{})},
	"evaluation.Point":            {typ: reflect.TypeOf(evaluation.Point{})},
	"acceptance.ProfileSet":       {typ: reflect.TypeOf(acceptance.ProfileSet{})},
	"acceptance.Profile":          {typ: reflect.TypeOf(acceptance.Profile{})},
	"acceptance.Provenance":       {typ: reflect.TypeOf(acceptance.Provenance{})},
	"acceptance.Sampling":         {typ: reflect.TypeOf(acceptance.Sampling{})},
	"acceptance.Value":            {typ: reflect.TypeOf(acceptance.Value{})},
	"acceptance.Pin":              {typ: reflect.TypeOf(acceptance.Pin{})},
	"routing.ProfileSet":          {typ: reflect.TypeOf(routing.ProfileSet{})},
	"routing.Profile":             {typ: reflect.TypeOf(routing.Profile{})},
	"routing.Point":               {typ: reflect.TypeOf(routing.Point{})},
	"routing.Provenance":          {typ: reflect.TypeOf(routing.Provenance{})},
	"routing.Value":               {typ: reflect.TypeOf(routing.Value{})},
	"routing.Pin":                 {typ: reflect.TypeOf(routing.Pin{})},
	"measurement.MeasuredRun":     {typ: reflect.TypeOf(measurement.MeasuredRun{})},
	"measurement.Source":          {typ: reflect.TypeOf(measurement.Source{})},
	"measurement.Adapter":         {typ: reflect.TypeOf(measurement.Adapter{})},
	"measurement.Input":           {typ: reflect.TypeOf(measurement.Input{})},
	"measurement.Extraction":      {typ: reflect.TypeOf(measurement.Extraction{})},
	"measurement.ClientSpec":      {typ: reflect.TypeOf(measurement.ClientSpec{})},
	"measurement.RampUp":          {typ: reflect.TypeOf(measurement.RampUp{})},
	"measurement.Tokenizer":       {typ: reflect.TypeOf(measurement.Tokenizer{})},
	"measurement.MetricDef":       {typ: reflect.TypeOf(measurement.MetricDef{})},
	"measurement.Point":           {typ: reflect.TypeOf(measurement.Point{})},
	"measurement.Load":            {typ: reflect.TypeOf(measurement.Load{})},
	"measurement.Value":           {typ: reflect.TypeOf(measurement.Value{})},
	"measurement.Realized":        {typ: reflect.TypeOf(measurement.Realized{})},
	"measurement.Noise":           {typ: reflect.TypeOf(measurement.Noise{})},
	"measurement.ProvenanceRow":   {typ: reflect.TypeOf(measurement.ProvenanceRow{})},
	"measurement.Observed":        {typ: reflect.TypeOf(measurement.Observed{})},
	"measurement.DiffEntry":       {typ: reflect.TypeOf(measurement.DiffEntry{})},
	"measurement.Registry":        {typ: reflect.TypeOf(measurement.Registry{})},
	"measurement.License":         {typ: reflect.TypeOf(measurement.License{})},
	"measurement.Evidence":        {typ: reflect.TypeOf(measurement.Evidence{})},
	"router.Router":               {typ: reflect.TypeOf(router.Router{})},
	"connector.Registry":          {typ: reflect.TypeOf(connector.Registry{})},
	"connector.Entry":             {typ: reflect.TypeOf(connector.Entry{})},
	"router.EPP":                  {typ: reflect.TypeOf(router.EPP{})},
	"router.ConfigRef":            {typ: reflect.TypeOf(router.ConfigRef{})},
	"fields.FieldList":            {typ: reflect.TypeOf(fields.FieldList{})},
	"fields.Engine":               {typ: reflect.TypeOf(fields.Engine{})},
	"fields.Field":                {typ: reflect.TypeOf(fields.Field{})},
	"fields.Evidence":             {typ: reflect.TypeOf(fields.Evidence{})},
	"fields.Estimation":           {typ: reflect.TypeOf(fields.Estimation{})},
	"pointrecord.Record":          {typ: reflect.TypeOf(pointrecord.Record{})},
	"pointrecord.Identity":        {typ: reflect.TypeOf(pointrecord.Identity{})},
	"pointrecord.G5Reason":        {typ: reflect.TypeOf(pointrecord.G5Reason{})},
	"pointrecord.Substitution":    {typ: reflect.TypeOf(pointrecord.Substitution{})},
	"pointrecord.Build":           {typ: reflect.TypeOf(pointrecord.Build{})},
	"pointrecord.Simulator":       {typ: reflect.TypeOf(pointrecord.Simulator{})},
	"pointrecord.Figure":          {typ: reflect.TypeOf(pointrecord.Figure{})},
	"pointrecord.Interval":        {typ: reflect.TypeOf(pointrecord.Interval{})},
	"pointrecord.Setting":         {typ: reflect.TypeOf(pointrecord.Setting{})},
	"pointrecord.Resolved":        {typ: reflect.TypeOf(pointrecord.Resolved{})},
	"pointrecord.Replay":          {typ: reflect.TypeOf(pointrecord.Replay{})},
	"pointrecord.TrajectoryStart": {typ: reflect.TypeOf(pointrecord.TrajectoryStart{})},
	"pointrecord.Facets":          {typ: reflect.TypeOf(pointrecord.Facets{})},
	"pointrecord.Parallelism":     {typ: reflect.TypeOf(pointrecord.Parallelism{})},
	"pointrecord.Note":            {typ: reflect.TypeOf(pointrecord.Note{})},
	"result.Result":               {typ: reflect.TypeOf(result.Result{})},
	"result.Engine":               {typ: reflect.TypeOf(result.Engine{})},
	"result.Data":                 {typ: reflect.TypeOf(result.Data{})},
	"result.DataSource":           {typ: reflect.TypeOf(result.DataSource{})},
	"result.Overlay":              {typ: reflect.TypeOf(result.Overlay{})},
	"result.SelectedEntries":      {typ: reflect.TypeOf(result.SelectedEntries{})},
	"result.SelectedCoefficient":  {typ: reflect.TypeOf(result.SelectedCoefficient{})},
	"result.SelectedProfile":      {typ: reflect.TypeOf(result.SelectedProfile{})},
	"result.Input":                {typ: reflect.TypeOf(result.Input{})},
	"result.Option":               {typ: reflect.TypeOf(result.Option{})},
	"result.WorkloadProperty":     {typ: reflect.TypeOf(result.WorkloadProperty{})},
	"result.Frame":                {typ: reflect.TypeOf(result.Frame{})},
	"result.Generator":            {typ: reflect.TypeOf(result.Generator{})},
	"result.InitialState":         {typ: reflect.TypeOf(result.InitialState{})},
	"result.Phases":               {typ: reflect.TypeOf(result.Phases{})},
	"result.WarmupPhase":          {typ: reflect.TypeOf(result.WarmupPhase{})},
	"result.MeasuredPhase":        {typ: reflect.TypeOf(result.MeasuredPhase{})},
	"result.DrainPhase":           {typ: reflect.TypeOf(result.DrainPhase{})},
	"result.StopRule":             {typ: reflect.TypeOf(result.StopRule{})},
	"result.AcceptanceCondition":  {typ: reflect.TypeOf(result.AcceptanceCondition{})},
	"result.Revisions":            {typ: reflect.TypeOf(result.Revisions{})},
	"result.Reference":            {typ: reflect.TypeOf(result.Reference{})},
	"result.RevisionChange":       {typ: reflect.TypeOf(result.RevisionChange{})},
	"result.Window":               {typ: reflect.TypeOf(result.Window{})},
	"result.Figure":               {typ: reflect.TypeOf(result.Figure{})},
	"result.Requests":             {typ: reflect.TypeOf(result.Requests{})},
	"result.RequestRow":           {typ: reflect.TypeOf(result.RequestRow{})},
	"result.AgenticTag":           {typ: reflect.TypeOf(result.AgenticTag{})},
	"result.Goodput":              {typ: reflect.TypeOf(result.Goodput{})},
	"result.ClassGoodput":         {typ: reflect.TypeOf(result.ClassGoodput{})},
	"result.Breakdowns":           {typ: reflect.TypeOf(result.Breakdowns{})},
	"result.Reports":              {typ: reflect.TypeOf(result.Reports{})},
	"result.Override":             {typ: reflect.TypeOf(result.Override{})},
	"result.Artifact":             {typ: reflect.TypeOf(result.Artifact{})},
	"result.Runtime":              {typ: reflect.TypeOf(result.Runtime{})},
}

var (
	fieldsMarker = regexp.MustCompile(`^<!-- fields: (\S+) -->$`)
	firstCode    = regexp.MustCompile("`([^`]+)`")
)

func TestDocFieldTablesMatchTypes(t *testing.T) {
	tables := map[string]map[string]bool{} // marker -> keys listed
	paths, err := filepath.Glob(filepath.Join("..", "docs", "contracts", "reference", "*.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		current := ""
		inTable := false // whether the current marker's table has begun
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if m := fieldsMarker.FindStringSubmatch(line); m != nil {
				current = m[1]
				if _, ok := documented[current]; !ok {
					t.Errorf("%s: marker names %q, which documented does not list", path, current)
				}
				if tables[current] == nil {
					tables[current] = map[string]bool{}
				}
				inTable = false
				continue
			}
			if current == "" {
				continue
			}
			if line == "" && !inTable {
				continue // between the marker and its table
			}
			if !strings.HasPrefix(line, "|") {
				current = ""
				continue
			}
			inTable = true
			cells := strings.Split(strings.Trim(line, "|"), "|")
			m := firstCode.FindStringSubmatch(cells[0])
			if m == nil {
				continue // the header row or the separator
			}
			tables[current][m[1]] = true
		}
		f.Close()
		if err := sc.Err(); err != nil {
			t.Fatal(err)
		}
	}

	names := make([]string, 0, len(documented))
	for n := range documented {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, name := range names {
		d := documented[name]
		got, ok := tables[name]
		if !ok {
			t.Errorf("%s: no field table in docs/reference/ (add one after a `<!-- fields: %s -->` line)", name, name)
			continue
		}
		want := yamlKeys(d.typ)
		for _, k := range d.omit {
			delete(want, k)
		}
		for k := range want {
			if !got[k] {
				t.Errorf("%s: key %q is missing from its field table", name, k)
			}
		}
		for k := range got {
			if !want[k] {
				t.Errorf("%s: field table lists %q, which the type does not declare", name, k)
			}
		}
	}
}

// yamlKeys returns the keys a type reads from YAML: every exported field's tag name,
// skipping fields tagged "-", flattening ",inline" fields, and using the lowercased Go name
// where a field has no tag.
func yamlKeys(t reflect.Type) map[string]bool {
	out := map[string]bool{}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("yaml")
		if tag == "-" || !f.IsExported() {
			continue
		}
		if strings.Contains(tag, ",inline") {
			for k := range yamlKeys(f.Type) {
				out[k] = true
			}
			continue
		}
		name := strings.Split(tag, ",")[0]
		if name == "" {
			name = strings.ToLower(t.Field(i).Name)
		}
		out[name] = true
	}
	return out
}

// TestDocEveryDocumentTypeHasATable walks every struct type reachable from the document
// roots and requires each to be listed in documented, so a nested type added to a schema
// cannot escape TestDocFieldTablesMatchTypes by never being named there.
func TestDocEveryDocumentTypeHasATable(t *testing.T) {
	roots := []reflect.Type{
		reflect.TypeOf(scenario.Scenario{}), reflect.TypeOf(deployment.Deployment{}),
		reflect.TypeOf(model.Graph{}), reflect.TypeOf(model.Identity{}),
		reflect.TypeOf(hardware.Chip{}), reflect.TypeOf(hardware.Fabric{}),
		reflect.TypeOf(hardware.StorageDevice{}), reflect.TypeOf(coefficient.Set{}),
		reflect.TypeOf(acceptance.ProfileSet{}),
		reflect.TypeOf(routing.ProfileSet{}),
		reflect.TypeOf(workload.Shape{}), reflect.TypeOf(evaluation.Run{}),
		reflect.TypeOf(measurement.MeasuredRun{}), reflect.TypeOf(measurement.Registry{}),
		reflect.TypeOf(connector.Registry{}),
		reflect.TypeOf(result.Result{}),
		reflect.TypeOf(fields.FieldList{}),
	}
	// Types whose wire form is not a mapping of this schema's keys, so they have no key table:
	// an interval, and a pool's engine block, which is the engine model's and held unopened.
	notMappings := map[reflect.Type]bool{reflect.TypeOf(coefficient.Interval{}): true,
		reflect.TypeOf(deployment.EngineBlock{}): true}

	have := map[reflect.Type]bool{}
	for _, d := range documented {
		have[d.typ] = true
	}
	seen := map[reflect.Type]bool{}
	var walk func(reflect.Type)
	walk = func(ty reflect.Type) {
		for ty.Kind() == reflect.Pointer || ty.Kind() == reflect.Slice ||
			ty.Kind() == reflect.Array || ty.Kind() == reflect.Map {
			ty = ty.Elem()
		}
		if ty.Kind() != reflect.Struct || seen[ty] ||
			!strings.HasPrefix(ty.PkgPath(), "github.com/inference-sim/blis-substrate/contracts/spec/") {
			return
		}
		seen[ty] = true
		if !have[ty] && !notMappings[ty] {
			t.Errorf("%s is reachable from a document but has no entry in documented and no key table", ty)
		}
		for i := 0; i < ty.NumField(); i++ {
			if f := ty.Field(i); f.IsExported() && f.Tag.Get("yaml") != "-" {
				walk(f.Type)
			}
		}
	}
	for _, r := range roots {
		walk(r)
	}
}

// keyList returns a type's YAML keys, for a table that leaves out an inlined type's keys
// documented in that type's own table.
func keyList(t reflect.Type) []string {
	var out []string
	for k := range yamlKeys(t) {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
