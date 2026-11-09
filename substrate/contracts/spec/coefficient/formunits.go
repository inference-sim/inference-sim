package coefficient

import (
	"regexp"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// FormUnits is the unit each coefficient family must have: the unit the formula that reads it
// requires (design SUB-16). The cost model reads a coefficient by name and multiplies it by a
// size or a count, so an entry in another unit prices wrongly without any error; this table
// is what makes the mismatch a load-time refusal. A family is matched by the name's prefix up
// to a variant suffix (a data type, a hardware part, a rank count, a backend).
//
// The table is the one place the unit of a family is written. The registry validator and the
// cost model both read it, so they cannot disagree about what a formula needs.
var FormUnits = []FormUnit{
	{`^activation_buffer_count_`, vocab.UnitDimensionless, "a count of buffers, a multiplier on the activation size"},
	{`^activation_scratch_floor_bytes$`, vocab.UnitBytesPerRank, "bytes on one rank"},
	{`^attention_(decode|prefill)_floor(_|$)`, vocab.UnitMicrosecondsPerXfer, "the launch floor of one kernel call, in microseconds"},
	{`^attention_(decode|prefill)_launch_floor_`, vocab.UnitMicrosecondsPerXfer, "the launch floor of one kernel call, in microseconds"},
	{`^attention_(decode|prefill)_split_floor_`, vocab.UnitMicrosecondsPerXfer, "the launch floor of one kernel call, in microseconds"},
	{`^attention_(decode|prefill)_(throughput_|serial_)?rate(_|$)`, vocab.UnitBytesPerMicrosecond, "a byte rate; the form divides KV bytes by it"},
	{`^attention_(decode|prefill)_per_token_`, vocab.UnitMicrosecondsPerToken, "microseconds per token, times a token count"},
	{`^attention_prefill_per_request_`, vocab.UnitMicrosecondsPerRequest, "microseconds per request, times a request count"},
	{`^attention_prefill_work_scale$`, vocab.UnitDimensionless, "a multiplier on the prefill work"},
	{`^collective_floor_`, vocab.UnitMicrosecondsPerXfer, "the floor of one collective call, in microseconds"},
	{`^collective_(peak|transition)_rate_`, vocab.UnitBytesPerMicrosecond, "a byte rate; the form divides message bytes by it"},
	{`^cudagraph_capture_`, vocab.UnitBytesPerRank, "graph-pool bytes on one rank"},
	{`^engine_workspace_bytes$`, vocab.UnitBytesPerRank, "bytes on one rank"},
	{`^nccl_communicator_bytes_`, vocab.UnitBytesPerRank, "bytes on one rank"},
	{`^fabric_bandwidth_efficiency$`, vocab.UnitDimensionless, "an efficiency, a fraction of the fabric's peak"},
	{`^gemm_eps_max_`, vocab.UnitDimensionless, "an efficiency asymptote, a fraction of the peak"},
	{`^gemm_(m|n|k)_half_`, vocab.UnitTokens, "the size at which half the asymptote is reached, in the dimension's elements"},
	{`^hbm_derate$`, vocab.UnitDimensionless, "a fraction of the datasheet bandwidth"},
	{`^host_admission_per_request$`, vocab.UnitMicrosecondsPerRequest, "microseconds per request"},
	{`^host_admission_per_token(_|$)`, vocab.UnitMicrosecondsPerToken, "microseconds per prompt token"},
	{`^host_completion$`, vocab.UnitMicrosecondsPerRequest, "microseconds per request"},
	{`^host_gather_rate_uva$`, vocab.UnitBytesPerMicrosecond, "a byte rate"},
	{`^host_launch_(eager_)?per_kernel$`, vocab.UnitMicrosecondsPerXfer, "microseconds per kernel launch"},
	{`^host_launch_eager_per_layer$`, vocab.UnitMicrosecondsPerLayer, "microseconds per layer"},
	{`^host_link_bandwidth$`, vocab.UnitBytesPerMicrosecond, "a byte rate"},
	{`^host_output_token$`, vocab.UnitMicrosecondsPerToken, "microseconds per emitted token"},
	{`^host_replay_graph_per_step$`, vocab.UnitMicrosecondsPerStep, "microseconds per replayed step"},
	{`^host_step_per_request$`, vocab.UnitMicrosecondsPerRequest, "microseconds per scheduled request"},
	{`^mhc_floor_`, vocab.UnitMicrosecondsPerXfer, "the floor of one kernel call, in microseconds"},
	{`^mhc_rate_`, vocab.UnitBytesPerMicrosecond, "a byte rate"},
	{`^moe_a2a_floor_`, vocab.UnitMicrosecondsPerXfer, "the floor of one all-to-all call, in microseconds"},
	{`^moe_a2a_rate_`, vocab.UnitBytesPerMicrosecond, "a byte rate"},
	{`^moe_routing_imbalance_median$`, vocab.UnitDimensionless, "a ratio of the busiest rank's load to the mean"},
	{`^recurrent_(decode|prefill)_floor_`, vocab.UnitMicrosecondsPerXfer, "the floor of one scan call, in microseconds"},
	{`^recurrent_(decode|prefill)_rate_`, vocab.UnitTokens, "tokens per microsecond past the floor; the form divides a token count by it"},
}

// FormUnit is one row of FormUnits: names matching Pattern must carry Unit.
type FormUnit struct {
	Pattern string
	Unit    vocab.Unit
	Why     string // what the formula does with the value
}

var formUnitRes = func() []*regexp.Regexp {
	out := make([]*regexp.Regexp, len(FormUnits))
	for i, f := range FormUnits {
		out[i] = regexp.MustCompile(f.Pattern)
	}
	return out
}()

// RequiredUnit returns the unit a formula requires of the coefficient name, and false when no
// family in FormUnits covers the name.
func RequiredUnit(name string) (FormUnit, bool) {
	for i, re := range formUnitRes {
		if re.MatchString(name) {
			return FormUnits[i], true
		}
	}
	return FormUnit{}, false
}
