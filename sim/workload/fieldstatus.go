package workload

// The status of every field of a TraceV2 trace, as `blis replay` reads it (design SIM-11): each
// is consumed, refused or inert, and says which. A consumed field changes some byte of the
// Result when it changes; cmd's TestSIM11_TraceFieldsConsumedChangeTheResult toggles each one and
// checks. A refused value is refused by package check, the same validator its kind meets on the
// command line. An inert field is read and recorded, or carried to a re-exported trace, and
// changes nothing the simulation computes; the reason is stated.
//
// TestSIM11_EveryTraceFieldHasAStatus walks TraceHeader, TraceRouterConfig, TraceKVOffloadConfig,
// TraceKVOffloadTier and TraceRecord with reflect and fails on a field these tables do not name,
// and on a name that is no field.

// FieldStatus is how replay treats one trace field.
type FieldStatus string

const (
	// FieldConsumed changes the simulation, so some byte of the Result, when it changes.
	FieldConsumed FieldStatus = "consumed"
	// FieldRefused is refused whenever it holds a value the simulator cannot honour.
	FieldRefused FieldStatus = "refused"
	// FieldInert is read but changes nothing the simulation computes.
	FieldInert FieldStatus = "inert"
)

// FieldEntry is one field's status and the reason for it.
type FieldEntry struct {
	Status FieldStatus
	Why    string
}

// traceHeaderFields is TraceHeader's table, keyed by Go field name.
var traceHeaderFields = map[string]FieldEntry{
	"Version":              {FieldRefused, "a version other than 4 is refused; version 3 names `blis convert trace`. The CSV layout is read from the column header row"},
	"TimeUnit":             {FieldInert, "recorded in the Result's input; every timestamp is read as microseconds whatever it states (writers write \"microseconds\")"},
	"Mode":                 {FieldConsumed, "recorded in the Result's input; mode converted lets --kv-offload-config add a deployment the trace does not record (RecordsDeployment); another mode is refused"},
	"WorkloadSeed":         {FieldInert, "recorded in the Result's input; replay draws its token IDs from --seed, not from the seed that generated the trace"},
	"SessionContextGrowth": {FieldConsumed, "accumulate builds each round's input on the previous round's; an unknown value is refused by the session loaders"},
	"GoodputSLOTargets":    {FieldConsumed, "the goodput targets when no --slo-* flag states them; a non-positive or non-finite target is refused (check.LatencyTargetMs)"},
	"KVOffload":            {FieldConsumed, "the resolved offload configuration the run used; replay rebuilds it, and refuses one it cannot reproduce"},
	"MaxNodesSpanned":      {FieldRefused, "a value above 1 is refused: replay cannot rebuild a multi-node fleet (#1530); negative is refused (check.Count); 0 and 1 change nothing"},
	"Router":               {FieldConsumed, "the router the run routed with; replay rebuilds it (INV-13)"},
}

// traceRouterFields is TraceRouterConfig's table.
var traceRouterFields = map[string]FieldEntry{
	"EPPFlags":         {FieldConsumed, "the Endpoint Picker's flags, with its configuration inlined"},
	"Manifests":        {FieldConsumed, "the Kubernetes objects the router reads (InferencePool, InferenceObjective)"},
	"PodLabels":        {FieldConsumed, "labels a configuration's selectors and filters read"},
	"GatewayLatencyUs": {FieldConsumed, "delay before the router first sees a request; negative is refused (check.DurationUs, as --gateway-latency)"},
	"ScrapeJitterUs":   {FieldConsumed, "phase offset of each endpoint's metrics scrape; negative is refused (check.DurationUs, as --epp-scrape-jitter)"},
	"KVEventDelayUs":   {FieldConsumed, "delay of KV events to the precise prefix producer; negative is refused (check.DurationUs, as --kv-event-delay)"},
	"Seed":             {FieldConsumed, "the router's random streams"},
}

// traceKVOffloadFields and traceKVOffloadTierFields are the offload block's tables. The block is
// replayed whole: build.ReconcileReplayKVOffload rebuilds the run's offload store from it and
// refuses a configuration this build cannot reproduce. A changed field changes the store; the
// cmd tests that pin this are TestKVOffload_EndToEnd_RunReplayRoundTrip and
// TestINV13_RunReplayParity_Offload. The reflection test checks every field is named here; it does
// not toggle each one (the block's own tests cover it).
var traceKVOffloadFields = map[string]FieldEntry{
	"CPUBytesToUse":          {FieldConsumed, "the CPU tier's size"},
	"PerBlockBytes":          {FieldConsumed, "the bytes of one offloaded block"},
	"BlockSize":              {FieldConsumed, "tokens per offload block"},
	"BlocksPerChunk":         {FieldConsumed, "blocks per offload chunk"},
	"TokensPerHash":          {FieldConsumed, "tokens per block hash"},
	"EvictionPolicy":         {FieldConsumed, "the CPU tier's eviction order"},
	"OffloadPromptOnly":      {FieldConsumed, "whether output blocks are offloaded"},
	"SelfDescribingKVEvents": {FieldConsumed, "the KV event encoding the router receives"},
	"Tiers":                  {FieldConsumed, "the secondary tiers below the CPU tier"},
	"SimpleCPUOffload":       {FieldConsumed, "vLLM's SimpleCPUOffloadConnector in place of the tiered one"},
	"LazyOffload":            {FieldConsumed, "the simple connector's lazy mode"},
}

var traceKVOffloadTierFields = map[string]FieldEntry{
	"Type":                   {FieldConsumed, "the tier's kind; one this build cannot reproduce is refused"},
	"RootDir":                {FieldInert, "a file-system tier's directory on the measured host; nothing in the simulation reads it"},
	"NReadThreads":           {FieldConsumed, "concurrent reads the tier serves"},
	"NWriteThreads":          {FieldConsumed, "concurrent writes the tier serves"},
	"Locality":               {FieldConsumed, "where the tier sits relative to the engine"},
	"EnableKVEvents":         {FieldConsumed, "whether the tier publishes KV events"},
	"DirectIO":               {FieldConsumed, "the tier's I/O path"},
	"DeviceClass":            {FieldInert, "the catalog device the bandwidths were resolved from; replay uses the resolved numbers"},
	"ReadBandwidth":          {FieldConsumed, "resolved read bandwidth"},
	"WriteBandwidth":         {FieldConsumed, "resolved write bandwidth"},
	"BaseLatency":            {FieldConsumed, "resolved per-transfer latency"},
	"SaturationQueueDepth":   {FieldConsumed, "queue depth at which bandwidth saturates"},
	"SingleTransferFraction": {FieldConsumed, "bandwidth fraction one transfer reaches"},
	"CapacityBytes":          {FieldConsumed, "the tier's size"},
	"LatencyJitterStddev":    {FieldConsumed, "the tier's latency jitter"},
}

// traceRecordFields is TraceRecord's table: one CSV row. Fields that only a router configuration,
// an encode pool or a session mode reads are consumed under that configuration, and the
// reflection test toggles them under it.
var traceRecordFields = map[string]FieldEntry{
	"RequestID":        {FieldInert, "names the request (request_<id>) and orders a re-export; no figure depends on the name"},
	"ClientID":         {FieldInert, "carried to the request and to a re-export; neither the router nor the engine reads it"},
	"TenantID":         {FieldConsumed, "the router's fairness identifier, and the Result's per-tenant breakdown"},
	"SLOClass":         {FieldConsumed, "the router's objective and the goodput class"},
	"SessionID":        {FieldConsumed, "groups rounds into sessions (closed-loop modes) and the session breakdown"},
	"RoundIndex":       {FieldConsumed, "orders a session's rounds; negative is refused (check.Count)"},
	"PrefixGroup":      {FieldConsumed, "requests of one group share their first prefix_length tokens"},
	"PrefixLength":     {FieldConsumed, "the shared prefix's length, for a request with a prefix_group; negative is refused (check.TokenCount)"},
	"Streaming":        {FieldConsumed, "the response is streamed, which the router sees through its response headers"},
	"InputTokens":      {FieldConsumed, "the prompt's length; negative is refused (check.TokenCount); 0 is dropped as unservable at the gateway"},
	"OutputTokens":     {FieldConsumed, "the output budget; negative is refused (check.TokenCount)"},
	"TextTokens":       {FieldInert, "the text share of the prompt; only image, audio and video counts are read"},
	"ImageTokens":      {FieldConsumed, "makes the request multimodal: a chat request, placed on an encode pod; negative is refused"},
	"AudioTokens":      {FieldConsumed, "as image_tokens"},
	"VideoTokens":      {FieldConsumed, "as image_tokens"},
	"ReasonRatio":      {FieldInert, "carried to the request; no part of the simulation reads it; outside [0, 1] is refused (check.Fraction)"},
	"Model":            {FieldConsumed, "the model the request names, and the per-model breakdown"},
	"DeadlineUs":       {FieldConsumed, "the client deadline; negative is refused (check.DurationUs); 0 is none"},
	"SLOTargetUs":      {FieldConsumed, "the TTFT target the router reads (x-llm-d-slo-ttft-ms); negative is refused"},
	"ArrivalTimeUs":    {FieldConsumed, "when the request arrives; negative is refused (check.DurationUs)"},
	"FirstChunkTimeUs": {FieldInert, "a measured outcome, carried to a re-export"},
	"LastChunkTimeUs":  {FieldInert, "a measured outcome, carried to a re-export"},
	"Status":           {FieldInert, "a measured outcome, carried to a re-export"},
	"Adapter":          {FieldConsumed, "the LoRA adapter the request names"},
	"ThinkTimeUs":      {FieldConsumed, "closed-loop think time before the round; negative is refused (check.DurationUs)"},
	"InputTokensReset": {FieldConsumed, "accumulate replay re-seeds the session's context to this length; negative is refused"},
}

// TraceFieldStatus returns the table for one trace struct, keyed by its Go type name
// (TraceHeader, TraceRouterConfig, ...), and false for a type with no table. The returned map is a
// copy (R8).
func TraceFieldStatus(typeName string) (map[string]FieldEntry, bool) {
	var m map[string]FieldEntry
	switch typeName {
	// The trace types are contracts/spec/trace's, under their own names; each is accepted by
	// either name.
	case "TraceHeader", "Header":
		m = traceHeaderFields
	case "TraceRouterConfig", "Router":
		m = traceRouterFields
	case "TraceKVOffloadConfig", "KVOffload":
		m = traceKVOffloadFields
	case "TraceKVOffloadTier", "KVOffloadTier":
		m = traceKVOffloadTierFields
	case "TraceRecord", "Record":
		m = traceRecordFields
	default:
		return nil, false
	}
	out := make(map[string]FieldEntry, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out, true
}
