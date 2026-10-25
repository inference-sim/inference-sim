package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/llmd"
	_ "github.com/inference-sim/inference-sim/sim/vendored" // the adapter's vendoring root
	"github.com/inference-sim/inference-sim/sim/workload"
)

// Router inputs. A simulated deployment routes with the llm-d router, configured by the same
// three inputs a real deployment's router reads: its command-line flags, the
// EndpointPickerConfig they name, and the InferencePool / InferenceObjective /
// InferenceModelRewrite manifests, plus the labels on the model-server pods. They are a
// deployment property, so the --scenario states them: the deployment's `router:` block and each
// pool's `labels` (#1950), as `parallel:` states the tensor-parallel width. The flags below model
// the environment around the router, not the deployment.
var (
	gatewayLatencyUs int64 // --gateway-latency
	scrapeJitterUs   int64 // --epp-scrape-jitter
	kvEventDelayUs   int64 // --kv-event-delay

	// scenarioRouter is the deployment half of the router inputs, read from the --scenario by
	// adoptKernelDeployment; scenarioStatesRouter reports whether the scenario states any of it
	// (a router block or a pool with labels) rather than leaving the router at its default.
	scenarioRouter       routerInputs
	scenarioStatesRouter bool
)

// registerRouterFlags registers the router's environment flags on run and replay (INV-13).
func registerRouterFlags(cmd *cobra.Command) {
	cmd.Flags().Int64Var(&gatewayLatencyUs, "gateway-latency", 0, "Microseconds from a request's arrival to the router's first look at it (network and proxy hops before the Endpoint Picker)")
	cmd.Flags().Int64Var(&kvEventDelayUs, "kv-event-delay", 0, "Microseconds from an engine publishing its KV-cache events (at the end of a step, as vLLM does) to the router's precise-prefix-cache-producer ingesting them; 0 = before any later routing decision")
	cmd.Flags().Int64Var(&scrapeJitterUs, "epp-scrape-jitter", 0, "Upper bound (µs) of the seeded phase offset of each endpoint's metrics collection; 0 = every endpoint is scraped in phase")
}

// routerInputs is everything that configures the embedded router. It is recorded in the
// TraceV2 header so a replay rebuilds the same router (INV-13).
type routerInputs struct {
	EPPFlags         []string
	Manifests        string
	PodLabels        map[string]string
	GatewayLatencyUs int64
	ScrapeJitterUs   int64
	KVEventDelayUs   int64
	// Seed is the router seed a replay takes from its trace; nil derives it from the run seed.
	Seed *int64
}

// validateRouterFlags refuses invalid router timing flags. run and replay call it with their other
// flag checks, before any simulation state is built (R3).
func validateRouterFlags() {
	if gatewayLatencyUs < 0 {
		logrus.Fatalf("--gateway-latency must be >= 0, got %d", gatewayLatencyUs)
	}
	if scrapeJitterUs < 0 {
		logrus.Fatalf("--epp-scrape-jitter must be >= 0, got %d", scrapeJitterUs)
	}
	if kvEventDelayUs < 0 {
		logrus.Fatalf("--kv-event-delay must be >= 0, got %d", kvEventDelayUs)
	}
}

// adoptScenarioRouter reads the router half of the scenario's deployment, refusing a router block
// whose files cannot be read before any simulation state is built (R3). adoptKernelDeployment
// calls it with the scenario's Shape.
func adoptScenarioRouter(shape scenarioShapeT) {
	in, err := scenarioRouterInputs(shape)
	if err != nil {
		logrus.Fatalf("scenario %q: %v", kernelScenario, err)
	}
	scenarioRouter, scenarioStatesRouter = in, statesRouter(shape)
}

// resolveRouterInputs is the router inputs of a run: the deployment half the scenario states
// (scenarioRouter), and the environment half the command line gives.
func resolveRouterInputs() routerInputs {
	validateRouterFlags()
	in := scenarioRouter
	in.EPPFlags = append([]string(nil), in.EPPFlags...)
	if len(in.EPPFlags) == 0 {
		in.EPPFlags = append([]string(nil), llmd.DefaultEPPFlags()...)
	}
	in.GatewayLatencyUs, in.ScrapeJitterUs, in.KVEventDelayUs = gatewayLatencyUs, scrapeJitterUs, kvEventDelayUs
	return in
}

// statesRouter reports whether a scenario states any router input: a router block, or a pool
// with labels.
func statesRouter(sh scenarioShapeT) bool {
	if sh.Router != nil {
		return true
	}
	for _, l := range sh.Labels {
		if len(l) > 0 {
			return true
		}
	}
	return false
}

// scenarioRouterInputs maps a scenario's router block and pool labels onto the router inputs,
// exactly as a real deployment hands them to its router. One block configures one router:
//
//   - epp_flags are the router's command-line flags, verbatim; absent, BLIS runs the default
//     standalone router whose pool is every simulated pod (llmd.DefaultEPPFlags);
//   - endpoint_picker_config is read and passed as --config-text=<contents>, which is what
//     --config-file reads; a --config-file among the flags is inlined the same way, so the trace
//     header carries the configuration itself rather than a path;
//   - manifests are read and joined into one YAML stream;
//   - pools[].labels label the pods of each pool (poolPodLabels).
//
// Paths, in the block and in a --config-file flag, are relative to the scenario file's directory.
func scenarioRouterInputs(sh scenarioShapeT) (routerInputs, error) {
	var in routerInputs
	r := sh.Router
	if r == nil {
		r = &deployment.Router{}
	}
	resolve := func(p string) string {
		if filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(sh.Dir, p)
	}
	if len(r.EPPFlags) > 0 {
		in.EPPFlags = append([]string(nil), r.EPPFlags...)
	} else {
		in.EPPFlags = append([]string(nil), llmd.DefaultEPPFlags()...)
	}
	if r.EndpointPickerConfig != "" {
		for _, f := range in.EPPFlags {
			if strings.HasPrefix(f, "--config-file") || strings.HasPrefix(f, "--config-text") {
				return routerInputs{}, fmt.Errorf("router.endpoint_picker_config and the EPP flag %q both name the router configuration; give one", f)
			}
		}
		b, err := os.ReadFile(resolve(r.EndpointPickerConfig))
		if err != nil {
			return routerInputs{}, fmt.Errorf("router.endpoint_picker_config: %w", err)
		}
		in.EPPFlags = append(in.EPPFlags, "--config-text="+string(b))
	}
	flags, err := inlineConfigFile(in.EPPFlags, resolve)
	if err != nil {
		return routerInputs{}, err
	}
	in.EPPFlags = flags
	var docs []string
	for i, p := range r.Manifests {
		b, err := os.ReadFile(resolve(p))
		if err != nil {
			return routerInputs{}, fmt.Errorf("router.manifests[%d]: %w", i, err)
		}
		docs = append(docs, strings.TrimRight(string(b), "\n"))
	}
	in.Manifests = strings.Join(docs, "\n---\n")
	in.PodLabels = poolPodLabels(sh)
	return in, nil
}

// poolPodLabels maps each pool's labels onto the role-scoped pod labels podLabelFunc applies: a
// label every pool states with one value labels every pod ("key"); any other labels the pods of
// its pool's role ("prefill:key", "decode:key").
//
// llm-d.ai/role on a prefill or decode pool is BLIS's to set from the pool's role, and blis-schemas
// refuses a value that contradicts it, so restating it adds nothing and is skipped. A colocated
// pool's pods carry no role label of their own, so its labels -- llm-d.ai/role included, should
// the pool state it -- label every pod.
func poolPodLabels(sh scenarioShapeT) map[string]string {
	out := map[string]string{}
	everyPool := func(k, v string) bool {
		for _, l := range sh.Labels {
			if got, ok := l[k]; !ok || got != v {
				return false
			}
		}
		return true
	}
	for i, labels := range sh.Labels {
		role := sh.Roles[i]
		for k, v := range labels {
			switch {
			case role != deployment.RoleColocated && k == roleLabel:
				continue
			case role == deployment.RoleColocated || everyPool(k, v):
				out[k] = v
			default:
				out[role.RoleLabelValue()+":"+k] = v
			}
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// inlineConfigFile replaces --config-file=<path> with --config-text=<contents>, so the trace header
// carries the configuration itself rather than a path the replaying host may not have. resolve
// maps the flag's path to the file read.
func inlineConfigFile(flags []string, resolve func(string) string) ([]string, error) {
	out := make([]string, 0, len(flags))
	for i := 0; i < len(flags); i++ {
		f := flags[i]
		var path string
		switch {
		case strings.HasPrefix(f, "--config-file="):
			path = strings.TrimPrefix(f, "--config-file=")
		case f == "--config-file" && i+1 < len(flags):
			path = flags[i+1]
			i++
		default:
			out = append(out, f)
			continue
		}
		b, err := os.ReadFile(resolve(path))
		if err != nil {
			return nil, fmt.Errorf("router.epp_flags --config-file: %w", err)
		}
		out = append(out, "--config-text="+string(b))
	}
	return out, nil
}

// environmentFlagsChanged reports whether the command line names a router environment flag.
func environmentFlagsChanged(cmd *cobra.Command) bool {
	for _, name := range []string{"gateway-latency", "epp-scrape-jitter", "kv-event-delay"} {
		if cmd.Flags().Changed(name) {
			return true
		}
	}
	return false
}

// header is the TraceV2 header record of the router inputs and the seed the router ran with.
func (in routerInputs) header(seed int64) *workload.TraceRouterConfig {
	return &workload.TraceRouterConfig{
		EPPFlags:         in.EPPFlags,
		Manifests:        in.Manifests,
		PodLabels:        in.PodLabels,
		GatewayLatencyUs: in.GatewayLatencyUs,
		ScrapeJitterUs:   in.ScrapeJitterUs,
		KVEventDelayUs:   in.KVEventDelayUs,
		Seed:             &seed,
	}
}

// reconcileReplayRouter picks the router inputs of a replay: the trace header's, which reproduce
// the source run (INV-13). A replay may name each half of them again, and each is checked against
// the header rather than silently preferred: the deployment half when the --scenario states a
// router (a router block or pool labels), the environment half when a router environment flag is
// given. Either half contradicting the header is refused (logrus.Fatalf). A scenario that states no
// router replays the recorded one. With no header (a trace BLIS did not record) the scenario and
// the flags configure the router, as on a run.
func reconcileReplayRouter(cmd *cobra.Command, h *workload.TraceRouterConfig) routerInputs {
	resolved := resolveRouterInputs()
	if h == nil {
		return resolved
	}
	fromHeader := routerInputs{
		EPPFlags: h.EPPFlags, Manifests: h.Manifests, PodLabels: h.PodLabels,
		GatewayLatencyUs: h.GatewayLatencyUs, ScrapeJitterUs: h.ScrapeJitterUs, KVEventDelayUs: h.KVEventDelayUs, Seed: h.Seed,
	}
	if scenarioStatesRouter {
		got, want := normalizeRouter(resolved), normalizeRouter(fromHeader)
		if !reflect.DeepEqual(got.EPPFlags, want.EPPFlags) || got.Manifests != want.Manifests ||
			!reflect.DeepEqual(got.PodLabels, want.PodLabels) {
			logrus.Fatalf("scenario %q states a router (its router block or pool labels) that differs from the "+
				"router the trace was recorded with; replay with a scenario stating the recorded router, or none, "+
				"to replay the recorded router (INV-13)", kernelScenario)
		}
	}
	if environmentFlagsChanged(cmd) && (resolved.GatewayLatencyUs != h.GatewayLatencyUs ||
		resolved.ScrapeJitterUs != h.ScrapeJitterUs || resolved.KVEventDelayUs != h.KVEventDelayUs) {
		logrus.Fatalf("the router flags differ from the router the trace was recorded with; omit them to replay the recorded router (INV-13)")
	}
	return fromHeader
}

// normalizeRouter is the part of the router inputs a scenario or a command line can name: the seed
// comes from the trace alone, and an empty label set or flag list is none.
func normalizeRouter(in routerInputs) routerInputs {
	if len(in.PodLabels) == 0 {
		in.PodLabels = nil
	}
	if len(in.EPPFlags) == 0 {
		in.EPPFlags = nil
	}
	in.Seed = nil
	return in
}

// newGateway builds the embedded llm-d router for a cluster run.
func newGateway(in routerInputs, model string) *llmd.Gateway {
	return llmd.New(llmd.Options{
		EPPFlags:         in.EPPFlags,
		Manifests:        []byte(in.Manifests),
		Model:            model,
		PodLabels:        podLabelFunc(in.PodLabels),
		GatewayLatencyUs: in.GatewayLatencyUs,
		ScrapeJitter:     time.Duration(in.ScrapeJitterUs) * time.Microsecond,
		KVEventDelayUs:   in.KVEventDelayUs,
		Seed:             in.Seed,
		Logger:           routerLogger(),
	})
}

// roleLabel is the llm-d pod label naming a pod's P/D role.
const roleLabel = "llm-d.ai/role"

// podLabelFunc labels each pod with its llm-d role (llm-d.ai/role, from the scenario's pools) and
// the extra labels: an unscoped key=value on every pod, a role:key=value on the pods of that role,
// where it overrides an unscoped label with the same key.
func podLabelFunc(extra map[string]string) func(cluster.InstanceID, cluster.PoolRole) map[string]string {
	return func(_ cluster.InstanceID, role cluster.PoolRole) map[string]string {
		labels := map[string]string{}
		for k, v := range extra {
			if !strings.Contains(k, ":") {
				labels[k] = v
			}
		}
		name := ""
		switch role {
		case cluster.PoolRolePrefill:
			name = "prefill"
		case cluster.PoolRoleDecode:
			name = "decode"
		case cluster.PoolRolePrefillDecode:
			name = "prefill-decode"
		case cluster.PoolRoleEncode:
			name = "encode"
		}
		if name != "" {
			labels[roleLabel] = name
			for k, v := range extra {
				if r, key, ok := strings.Cut(k, ":"); ok && r == name {
					labels[key] = v
				}
			}
		}
		return labels
	}
}

// routerLogger routes the router's logs to stderr at BLIS's debug and trace levels; stdout
// carries only simulation results (INV-6).
func routerLogger() logr.Logger {
	verbosity := -1
	switch logrus.GetLevel() {
	case logrus.DebugLevel:
		verbosity = 2
	case logrus.TraceLevel:
		verbosity = 5
	}
	return logr.New(&routerLogSink{verbosity: verbosity})
}

// routerLogSink writes the router's errors to stderr as errors (shown at every --log level but
// fatal and panic) and its informational messages at BLIS's debug and trace levels.
type routerLogSink struct {
	verbosity int
	name      string
	values    []any
}

func (s *routerLogSink) Init(logr.RuntimeInfo)  {}
func (s *routerLogSink) Enabled(level int) bool { return level <= s.verbosity }

func (s *routerLogSink) Info(_ int, msg string, kv ...any) {
	logrus.Debug(s.format(msg, nil, kv))
}

func (s *routerLogSink) Error(err error, msg string, kv ...any) {
	logrus.Error(s.format(msg, err, kv))
}

func (s *routerLogSink) WithValues(kv ...any) logr.LogSink {
	c := *s
	c.values = append(append([]any(nil), s.values...), kv...)
	return &c
}

func (s *routerLogSink) WithName(name string) logr.LogSink {
	c := *s
	c.name = strings.TrimPrefix(s.name+"/"+name, "/")
	return &c
}

func (s *routerLogSink) format(msg string, err error, kv []any) string {
	var b strings.Builder
	b.WriteString("[llm-d] ")
	if s.name != "" {
		b.WriteString(s.name + ": ")
	}
	b.WriteString(msg)
	if err != nil {
		fmt.Fprintf(&b, " error=%q", err.Error())
	}
	all := append(append([]any(nil), s.values...), kv...)
	for i := 0; i+1 < len(all); i += 2 {
		fmt.Fprintf(&b, " %v=%v", all[i], all[i+1])
	}
	return b.String()
}
