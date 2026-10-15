package cmd

import (
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/llmd"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// Router inputs. A simulated deployment routes with the llm-d router, configured by the same
// three inputs a real deployment's router reads: its command-line flags, the
// EndpointPickerConfig they name, and the InferencePool / InferenceObjective /
// InferenceModelRewrite manifests. The remaining flags model the environment, not routing.
var (
	eppFlagsPath     string   // --epp-flags: file with the EPP command-line flags
	eppConfigPath    string   // --epp-config: EndpointPickerConfig file
	eppManifestPaths []string // --epp-manifests: Kubernetes manifest files
	podLabelPairs    []string // --pod-labels: extra labels on every model-server pod
	gatewayLatencyUs int64    // --gateway-latency
	scrapeJitterUs   int64    // --epp-scrape-jitter
)

// registerRouterFlags registers the router inputs on run and replay (INV-13).
func registerRouterFlags(cmd *cobra.Command) {
	cmd.Flags().StringVar(&eppFlagsPath, "epp-flags", "", "File holding the llm-d router's (EPP's) command-line flags, verbatim: a YAML list of flags, or a YAML map of flag name to value as in the router Helm chart's epp.flags. Default: a standalone router whose pool is every simulated pod ("+strings.Join(llmd.DefaultEPPFlags(), " ")+")")
	cmd.Flags().StringVar(&eppConfigPath, "epp-config", "", "EndpointPickerConfig file for the llm-d router, as passed to the EPP with --config-file. Default: the router's built-in default configuration")
	cmd.Flags().StringSliceVar(&eppManifestPaths, "epp-manifests", nil, "Kubernetes manifest files (comma-separated or repeated). The router reads InferencePool, InferenceObjective and InferenceModelRewrite objects; other kinds are ignored")
	cmd.Flags().StringSliceVar(&podLabelPairs, "pod-labels", nil, "Extra labels on the simulated model-server pods, so an InferencePool selector or label filter from a real deployment matches: key=value on every pod, or role:key=value on the pods of one P/D role (prefill, decode, prefill-decode, encode); comma-separated or repeated. llm-d.ai/role comes from the scenario's pools")
	cmd.Flags().Int64Var(&gatewayLatencyUs, "gateway-latency", 0, "Microseconds from a request's arrival to the router's first look at it (network and proxy hops before the Endpoint Picker)")
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
}

// resolveRouterInputs reads the router inputs the command line names.
func resolveRouterInputs() routerInputs {
	in := routerInputs{GatewayLatencyUs: gatewayLatencyUs, ScrapeJitterUs: scrapeJitterUs}
	validateRouterFlags()
	if eppFlagsPath != "" {
		flags, err := readEPPFlags(eppFlagsPath)
		if err != nil {
			logrus.Fatalf("--epp-flags: %v", err)
		}
		in.EPPFlags = flags
	} else {
		in.EPPFlags = append([]string(nil), llmd.DefaultEPPFlags()...)
	}
	if eppConfigPath != "" {
		for _, f := range in.EPPFlags {
			if strings.HasPrefix(f, "--config-file") || strings.HasPrefix(f, "--config-text") {
				logrus.Fatalf("--epp-config and the EPP flag %q both name the router configuration; give one", f)
			}
		}
		b, err := os.ReadFile(eppConfigPath)
		if err != nil {
			logrus.Fatalf("--epp-config: %v", err)
		}
		in.EPPFlags = append(in.EPPFlags, "--config-text="+string(b))
	}
	in.EPPFlags = inlineConfigFile(in.EPPFlags)
	var docs []string
	for _, p := range eppManifestPaths {
		b, err := os.ReadFile(p)
		if err != nil {
			logrus.Fatalf("--epp-manifests: %v", err)
		}
		docs = append(docs, strings.TrimRight(string(b), "\n"))
	}
	in.Manifests = strings.Join(docs, "\n---\n")
	if len(podLabelPairs) > 0 {
		in.PodLabels = map[string]string{}
		for _, kv := range podLabelPairs {
			k, v, ok := strings.Cut(kv, "=")
			if !ok || k == "" {
				logrus.Fatalf("--pod-labels: %q is not [role:]key=value", kv)
			}
			role, key, scoped := strings.Cut(k, ":")
			if !scoped {
				key = k
			} else if !podRoles[role] {
				logrus.Fatalf("--pod-labels: %q scopes a label to role %q; the roles are prefill, decode, prefill-decode and encode", kv, role)
			}
			if key == roleLabel {
				logrus.Fatalf("--pod-labels: %s is set by BLIS from the scenario's pools and cannot be given", roleLabel)
			}
			in.PodLabels[k] = v
		}
	}
	return in
}

// readEPPFlags reads a flags file: a YAML list of command-line arguments, or a YAML map of flag
// name to value (the router Helm chart's epp.flags shape).
func readEPPFlags(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var node yaml.Node
	if err := yaml.Unmarshal(b, &node); err != nil {
		return nil, err
	}
	if len(node.Content) == 0 {
		return nil, nil
	}
	switch root := node.Content[0]; root.Kind {
	case yaml.SequenceNode:
		var list []string
		if err := root.Decode(&list); err != nil {
			return nil, err
		}
		return list, nil
	case yaml.MappingNode:
		var m map[string]string
		if err := root.Decode(&m); err != nil {
			return nil, err
		}
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out := make([]string, 0, len(keys))
		for _, k := range keys {
			out = append(out, "--"+strings.TrimPrefix(k, "--")+"="+m[k])
		}
		return out, nil
	default:
		return nil, fmt.Errorf("%s: expected a YAML list of flags or a map of flag name to value", path)
	}
}

// inlineConfigFile replaces --config-file=<path> with --config-text=<contents>, so the trace
// header carries the configuration itself rather than a path the replaying host may not have.
func inlineConfigFile(flags []string) []string {
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
		b, err := os.ReadFile(path)
		if err != nil {
			logrus.Fatalf("EPP flag --config-file: %v", err)
		}
		out = append(out, "--config-text="+string(b))
	}
	return out
}

// routerFlagsChanged reports whether the command line names any router input.
func routerFlagsChanged(cmd *cobra.Command) bool {
	for _, name := range []string{"epp-flags", "epp-config", "epp-manifests", "pod-labels", "gateway-latency", "epp-scrape-jitter"} {
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
		Seed:             &seed,
	}
}

// reconcileReplayRouter picks the router inputs of a replay: the trace header's, which
// reproduce the source run (INV-13), unless the command line names its own. A command line
// that contradicts the header is refused, never silently preferred.
func reconcileReplayRouter(cmd *cobra.Command, h *workload.TraceRouterConfig) routerInputs {
	flags := resolveRouterInputs()
	if h == nil {
		return flags
	}
	fromHeader := routerInputs{
		EPPFlags: h.EPPFlags, Manifests: h.Manifests, PodLabels: h.PodLabels,
		GatewayLatencyUs: h.GatewayLatencyUs, ScrapeJitterUs: h.ScrapeJitterUs, Seed: h.Seed,
	}
	if !routerFlagsChanged(cmd) {
		return fromHeader
	}
	if !reflect.DeepEqual(normalizeRouter(flags), normalizeRouter(fromHeader)) {
		logrus.Fatalf("the router flags differ from the router the trace was recorded with; omit them to replay the recorded router (INV-13)")
	}
	return fromHeader
}

// normalizeRouter is the part of the router inputs a command line can name: the seed comes from
// the trace alone.
func normalizeRouter(in routerInputs) routerInputs {
	if len(in.PodLabels) == 0 {
		in.PodLabels = nil
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
		Seed:             in.Seed,
		Logger:           routerLogger(),
	})
}

// roleLabel is the llm-d pod label naming a pod's P/D role.
const roleLabel = "llm-d.ai/role"

// podRoles are the llm-d.ai/role values; --pod-labels may scope a label to one of them.
var podRoles = map[string]bool{"prefill": true, "decode": true, "prefill-decode": true, "encode": true}

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

// routerLogSink writes the router's errors to stderr at every log level (a router error is never
// silent) and its informational messages at BLIS's debug and trace levels.
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
	logrus.Warn(s.format(msg, err, kv))
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
