// queueing_model_analyzer.go implements QueueingModelAnalyzer, an Analyzer that uses
// queueing model parameter estimation (Nelder-Mead sliding window or EKF) to compute
// per-variant maximum throughput given SLO targets, then derives scale-up/down signals.
package cluster

import (
	"math"
	"sort"

	tunerconfig "github.com/llm-inferno/model-tuner/pkg/config"
	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/model-tuner/pkg/estimator"
	qanalyzer "github.com/llm-inferno/queue-analysis/pkg/analyzer"
	"github.com/sirupsen/logrus"
)

const (
	// DefaultMaxBatchSize is used when ReplicaMetrics.MaxBatchSize is zero.
	DefaultMaxBatchSize = 256
	// DefaultMaxQueueSize is the external queue depth passed to the queueing model.
	DefaultMaxQueueSize = 100
	// DefaultSLOMultiplier scales observed base latency to derive an observation-based SLO target.
	DefaultSLOMultiplier = 3.0
	// DefaultFallbackHeadroom scales raw observed latency when no fitted parameters are available.
	DefaultFallbackHeadroom = 1.5
)

// SLOTarget holds per-model SLO targets in milliseconds.
type SLOTarget struct {
	TargetTTFT float32 // ms
	TargetITL  float32 // ms
}

// QMConfig configures QueueingModelAnalyzer behaviour.
type QMConfig struct {
	// SLOTargets maps ModelID → explicit SLO target (ms). When present, overrides observation-based SLO derivation.
	SLOTargets map[string]SLOTarget

	// SLOMultiplier scales fitted base latency to derive SLO targets from model parameters.
	// Default: DefaultSLOMultiplier.
	SLOMultiplier float64

	// TuningEnabled enables parameter estimation. When false, the analyzer only uses SLOTargets.
	TuningEnabled bool

	// InitObs is the number of observations required before Nelder-Mead fitting begins.
	// Default: 5.
	InitObs int

	// UseSliding uses the SlidingWindowEstimator (re-fit every cycle) instead of the EKF.
	UseSliding bool

	// WindowSize is the observation window for SlidingWindowEstimator. Default: 20.
	WindowSize int

	// ResidualThreshold for outlier rejection in SlidingWindowEstimator. Default: 0.3.
	ResidualThreshold float64

	// InitFitThreshold: if the initial Nelder-Mead objective exceeds this value, fall back to EKF.
	// 0 disables the threshold check.
	InitFitThreshold float64

	// WarmUpCycles is the number of EKF update cycles during which the NIS gate is disabled.
	// 0 = NIS always active (no warm-up). Positive N = warm-up for N cycles.
	WarmUpCycles int
}

// modelVariantKey identifies a (model, variant) pair for per-variant parameter tracking.
type modelVariantKey struct {
	ModelID string
	Variant VariantSpec
}

// perVariantState holds the parameter estimation state for one (model, variant) pair.
type perVariantState struct {
	ie          *estimator.InitEstimator
	swe         *estimator.SlidingWindowEstimator
	tuner       *core.Tuner
	ekfFallback bool
	ekfUpdates  int
	alpha       float64
	beta        float64
	gamma       float64
}

// QueueingModelAnalyzer implements the Analyzer interface using queueing model parameter
// estimation to derive per-replica maximum throughput under SLO constraints.
type QueueingModelAnalyzer struct {
	cfg          QMConfig
	variantState map[modelVariantKey]*perVariantState
}

// NewQueueingModelAnalyzer constructs a QueueingModelAnalyzer with the given config.
// Zero-valued fields are replaced with defaults. Panics on invalid (negative) numeric fields (R3).
func NewQueueingModelAnalyzer(cfg QMConfig) *QueueingModelAnalyzer {
	if cfg.SLOMultiplier < 0 {
		panic("NewQueueingModelAnalyzer: SLOMultiplier must be >= 0")
	}
	if cfg.InitObs < 0 {
		panic("NewQueueingModelAnalyzer: InitObs must be >= 0")
	}
	if cfg.WindowSize < 0 {
		panic("NewQueueingModelAnalyzer: WindowSize must be >= 0")
	}
	if cfg.ResidualThreshold < 0 {
		panic("NewQueueingModelAnalyzer: ResidualThreshold must be >= 0")
	}
	if cfg.SLOMultiplier == 0 {
		cfg.SLOMultiplier = DefaultSLOMultiplier
	}
	if cfg.InitObs == 0 {
		cfg.InitObs = 5
	}
	if cfg.WindowSize == 0 {
		cfg.WindowSize = 20
	}
	if cfg.ResidualThreshold == 0 {
		cfg.ResidualThreshold = 0.3
	}
	return &QueueingModelAnalyzer{
		cfg:          cfg,
		variantState: make(map[modelVariantKey]*perVariantState),
	}
}

// Name implements Analyzer.
func (a *QueueingModelAnalyzer) Name() string { return "queueing-model" }

// Analyze implements Analyzer. It groups replicas by variant, runs parameter estimation
// for each (model, variant) pair, queries queue-analysis for the maximum RPS under the
// SLO target, and computes TotalSupply / TotalDemand / Required/SpareCapacity.
func (a *QueueingModelAnalyzer) Analyze(ms ModelSignals) AnalyzerResult {
	result := AnalyzerResult{ModelID: ms.ModelID}
	if len(ms.Replicas) == 0 {
		return result
	}

	// Group replicas by variant.
	type variantGroup struct{ replicas []ReplicaMetrics }
	groups := make(map[modelVariantKey]*variantGroup)
	for _, rm := range ms.Replicas {
		k := modelVariantKey{ModelID: ms.ModelID, Variant: rm.Variant}
		if groups[k] == nil {
			groups[k] = &variantGroup{}
		}
		groups[k].replicas = append(groups[k].replicas, rm)
	}
	keys := sortedModelVariantKeys(groups)

	// Phase 1: parameter estimation (if enabled).
	if a.cfg.TuningEnabled {
		for _, k := range keys {
			g := groups[k]
			state := a.getOrCreateState(k)
			for _, rm := range g.replicas {
				if rm.ITL <= 0 || rm.DispatchRate <= 0 || rm.AvgInTokens <= 0 || rm.AvgOutTokens <= 0 {
					continue
				}
				maxBatch := int(rm.MaxBatchSize)
				if maxBatch <= 0 {
					maxBatch = DefaultMaxBatchSize
				}
				// TTFT and ITL are in μs; environment expects ms.
				env := core.NewEnvironmentPrefillDecode(
					float32(rm.DispatchRate*60), // req/min
					0,                           // batchSize (0 = let model compute)
					0,                           // avgQueueTime (ms)
					maxBatch,
					float32(rm.AvgInTokens),
					float32(rm.AvgOutTokens),
					float32(rm.TTFT/1000), // μs → ms
					float32(rm.ITL/1000),  // μs → ms
				)
				env.MaxQueueSize = DefaultMaxQueueSize
				if !env.Valid() {
					continue
				}
				a.updateVariantParameters(state, env)
				break // one observation per cycle per variant (first valid replica)
			}
		}
	}

	// Phase 2: derive SLO target for this model.
	slo, ok := a.getSLOTarget(ms.ModelID, ms.Replicas)
	if !ok {
		return result
	}

	// Phase 3: compute supply/demand per variant.
	totalSupply, totalDemand := 0.0, 0.0
	var variantCapacities []VariantCapacity

	for _, k := range keys {
		g := groups[k]
		state := a.variantState[k]
		if state == nil || state.alpha <= 0 {
			continue
		}

		replicaCount := len(g.replicas)
		wm := computeWorkloadMetrics(g.replicas)
		maxBatchForQA := DefaultMaxBatchSize
		for _, rm := range g.replicas {
			if rm.MaxBatchSize > 0 {
				maxBatchForQA = int(rm.MaxBatchSize)
				break
			}
		}

		maxRPS := a.maxRPSFromQueueAnalysis(state, slo, wm, maxBatchForQA)
		if maxRPS <= 0 {
			continue
		}

		variantSupply := maxRPS * float64(replicaCount)
		variantDemand := 0.0
		for _, rm := range g.replicas {
			variantDemand += rm.DispatchRate
		}
		costPerReplica := 0.0
		if len(g.replicas) > 0 {
			costPerReplica = g.replicas[0].CostPerHour
		}

		variantCapacities = append(variantCapacities, VariantCapacity{
			Variant:        k.Variant,
			Supply:         variantSupply,
			Demand:         variantDemand,
			ReplicaCount:   replicaCount,
			CostPerReplica: costPerReplica,
		})
		totalSupply += variantSupply
		totalDemand += variantDemand
	}

	result.TotalSupply = totalSupply
	result.TotalDemand = totalDemand
	result.VariantCapacities = variantCapacities
	if totalSupply > 0 {
		result.Utilization = totalDemand / totalSupply
	}

	// Compute required/spare capacity.
	// SpareCapacity: demand is below (supply - 1 replica's worth), safe to remove.
	if totalSupply > 0 {
		totalReplicas := 0
		for _, vc := range variantCapacities {
			totalReplicas += vc.ReplicaCount
		}
		perReplicaSupply := 0.0
		if totalReplicas > 0 {
			perReplicaSupply = totalSupply / float64(totalReplicas)
		}
		safeSupply := totalSupply - perReplicaSupply
		if totalDemand > totalSupply {
			result.RequiredCapacity = totalDemand - totalSupply
		} else if totalDemand < safeSupply {
			result.SpareCapacity = safeSupply - totalDemand
		}
	} else if totalDemand > 0 {
		result.RequiredCapacity = totalDemand
	}
	return result
}

// getOrCreateState returns the per-variant state, creating it if absent.
func (a *QueueingModelAnalyzer) getOrCreateState(k modelVariantKey) *perVariantState {
	if st, ok := a.variantState[k]; ok {
		return st
	}
	st := &perVariantState{
		ie: estimator.NewInitEstimator(a.cfg.InitObs, false),
	}
	a.variantState[k] = st
	return st
}

// updateVariantParameters adds an observation and, once ready, fits/updates model parameters.
func (a *QueueingModelAnalyzer) updateVariantParameters(state *perVariantState, env *core.EnvironmentPrefillDecode) {
	state.ie.AddObservation(env)
	if !state.ie.IsReady() {
		return
	}

	if a.cfg.UseSliding && !state.ekfFallback {
		if state.swe == nil {
			// First time: create SWE, seed from IE, run initial fit.
			state.swe = estimator.NewSlidingWindowEstimator(a.cfg.WindowSize, 1, a.cfg.ResidualThreshold)
			state.swe.SeedFromEstimator(state.ie)
			if fitted, err := state.ie.Fit(); err == nil {
				if a.cfg.InitFitThreshold > 0 && state.ie.LastFitFuncValue() > a.cfg.InitFitThreshold {
					logrus.Infof("[qma] poor init fit (obj=%.4g > %.4g), activating EKF fallback",
						state.ie.LastFitFuncValue(), a.cfg.InitFitThreshold)
					state.ekfFallback = true
				} else {
					state.swe.SeedLastFit(fitted)
				}
			}
			if !state.ekfFallback {
				fitted, err := state.swe.Fit()
				if err == nil && len(fitted) == 3 && fitted[0] > 0 && fitted[1] > 0 && fitted[2] > 0 {
					state.alpha, state.beta, state.gamma = fitted[0], fitted[1], fitted[2]
				}
				return
			}
		} else {
			state.swe.AddObservation(env)
			if !state.swe.IsReady() {
				return
			}
			fitted, err := state.swe.Fit()
			if err == nil && len(fitted) == 3 && fitted[0] > 0 && fitted[1] > 0 && fitted[2] > 0 {
				state.alpha, state.beta, state.gamma = fitted[0], fitted[1], fitted[2]
			}
			return
		}
	}

	// EKF path (UseSliding=false or ekfFallback=true).
	if state.tuner == nil {
		var fitInitState []float64
		if fitted, err := state.ie.Fit(); err == nil {
			fitInitState = fitted
		}
		tuner, _, err := core.SetupTunerForQueueingModel(buildEKFConfig(fitInitState), env, "prefill-decode")
		if err != nil {
			logrus.Warnf("[qma] failed to create EKF tuner: %v", err)
			return
		}
		state.tuner = tuner
	}
	skipNIS := state.ekfUpdates < a.cfg.WarmUpCycles
	results, err := state.tuner.RunWithValidation(env, skipNIS)
	if err != nil || results == nil || results.ValidationFailed {
		return
	}
	state.alpha = float64(results.ServiceParms.Alpha)
	state.beta = float64(results.ServiceParms.Beta)
	state.gamma = float64(results.ServiceParms.Gamma)
	state.ekfUpdates++
}

// getSLOTarget returns the SLO target for a model, using (in priority order):
// 1. Explicit SLOTargets from config.
// 2. Derived from fitted model parameters using SLOMultiplier.
// 3. Fallback: headroom over observed TTFT/ITL.
func (a *QueueingModelAnalyzer) getSLOTarget(modelID string, replicas []ReplicaMetrics) (SLOTarget, bool) {
	// Priority 1: explicit config override.
	if t, ok := a.cfg.SLOTargets[modelID]; ok {
		return t, true
	}

	// Priority 2: derive from fitted parameters.
	wm := computeWorkloadMetrics(replicas)
	var bestTTFT, bestITL float64
	for _, k := range sortedModelVariantKeys(a.variantState) {
		if k.ModelID != modelID {
			continue
		}
		state := a.variantState[k]
		if state.alpha <= 0 {
			continue
		}
		k2 := a.cfg.SLOMultiplier
		α, β, γ := state.alpha, state.beta, state.gamma
		// Base TTFT: prefill time at single-token batch ≈ α + (β+γ)*n_in
		ttft := k2*α + (β+γ)*wm.AvgInTokens
		// Base ITL: decode time ≈ α + β + γ*(n_in + n_out/2)
		itl := k2*α + β + γ*(wm.AvgInTokens+(wm.AvgOutTokens+1)/2)
		if ttft > bestTTFT {
			bestTTFT = ttft
		}
		if itl > bestITL {
			bestITL = itl
		}
	}
	if bestTTFT > 0 && bestITL > 0 {
		return SLOTarget{TargetTTFT: float32(bestTTFT), TargetITL: float32(bestITL)}, true
	}

	// Priority 3: headroom over observed latency.
	ttftMs := wm.AvgTTFT / 1000 // μs → ms
	itlMs := wm.AvgITL / 1000
	if ttftMs > 0 && itlMs > 0 {
		ttftTarget := ttftMs * DefaultFallbackHeadroom
		itlTarget := itlMs * DefaultFallbackHeadroom
		if ttftTarget > 10_000 {
			ttftTarget = 10_000
		}
		if itlTarget > 500 {
			itlTarget = 500
		}
		return SLOTarget{TargetTTFT: float32(ttftTarget), TargetITL: float32(itlTarget)}, true
	}
	return SLOTarget{}, false
}

// maxRPSFromQueueAnalysis queries the queue-analysis model for the max RPS satisfying the SLO.
// Returns 0 on any error.
func (a *QueueingModelAnalyzer) maxRPSFromQueueAnalysis(state *perVariantState, slo SLOTarget, wm workloadMetrics, maxBatch int) float64 {
	if state.alpha <= 0 {
		return 0
	}
	qCfg := &qanalyzer.Configuration{
		MaxBatchSize: maxBatch,
		MaxQueueSize: DefaultMaxQueueSize,
		ServiceParms: &qanalyzer.ServiceParms{
			Alpha: float32(state.alpha),
			Beta:  float32(state.beta),
			Gamma: float32(state.gamma),
		},
	}
	reqSize := &qanalyzer.RequestSize{
		AvgInputTokens:  float32(wm.AvgInTokens),
		AvgOutputTokens: float32(wm.AvgOutTokens),
	}
	qa, err := qanalyzer.NewLLMQueueAnalyzer(qCfg, reqSize)
	if err != nil {
		return 0
	}
	targetRate, _, _, err := qa.Size(&qanalyzer.TargetPerf{
		TargetTTFT: slo.TargetTTFT,
		TargetITL:  slo.TargetITL,
	})
	if err != nil || targetRate == nil {
		return 0
	}
	maxRPS := float64(min32(targetRate.RateTargetTTFT, targetRate.RateTargetITL))
	if maxRPS < 0 {
		return 0
	}
	return maxRPS
}

// workloadMetrics holds aggregated workload statistics across replicas.
type workloadMetrics struct {
	AvgTTFT      float64 // weighted mean TTFT (μs)
	AvgITL       float64 // weighted mean ITL (μs)
	AvgInTokens  float64
	AvgOutTokens float64
	DispatchRate float64 // total across all replicas (req/s)
}

// computeWorkloadMetrics aggregates replica-level metrics into deployment-level workload stats.
// Weighting is by DispatchRate; replicas with zero DispatchRate contribute to DispatchRate sum
// but not to the weighted averages.
func computeWorkloadMetrics(replicas []ReplicaMetrics) workloadMetrics {
	var totalWeight, sumTTFT, sumITL, sumIn, sumOut, totalRate float64
	for _, rm := range replicas {
		totalRate += rm.DispatchRate
		if rm.DispatchRate <= 0 {
			continue
		}
		w := rm.DispatchRate
		totalWeight += w
		sumTTFT += rm.TTFT * w
		sumITL += rm.ITL * w
		sumIn += rm.AvgInTokens * w
		sumOut += rm.AvgOutTokens * w
	}
	if totalWeight == 0 {
		return workloadMetrics{DispatchRate: totalRate}
	}
	return workloadMetrics{
		AvgTTFT:      sumTTFT / totalWeight,
		AvgITL:       sumITL / totalWeight,
		AvgInTokens:  sumIn / totalWeight,
		AvgOutTokens: sumOut / totalWeight,
		DispatchRate: totalRate,
	}
}

// buildEKFConfig constructs a ConfigData from the fitted initial state (or fallback defaults).
func buildEKFConfig(fitInitState []float64) *tunerconfig.ConfigData {
	initState := fitInitState
	if len(initState) != 3 || initState[0] <= 0 || initState[1] <= 0 || initState[2] <= 0 {
		initState = []float64{7.47, 0.044, 3.37e-5}
	}
	factor := tunerconfig.DefaultInitStateFactor
	eps := tunerconfig.DefaultInitStateMinEpsilon
	return &tunerconfig.ConfigData{
		FilterData: tunerconfig.FilterData{
			GammaFactor: float64(tunerconfig.DefaultGammaFactor),
			ErrorLevel:  float64(tunerconfig.DefaultErrorLevel),
			TPercentile: float64(tunerconfig.DefaultStudentPercentile),
		},
		ModelData: tunerconfig.ModelData{
			InitState: append([]float64(nil), initState...),
			PercentChange: []float64{
				float64(tunerconfig.DefaultPercentChange),
				float64(tunerconfig.DefaultPercentChange),
				float64(tunerconfig.DefaultPercentChange),
			},
			BoundedState: true,
			MinState: []float64{
				math.Max(initState[0]/factor, eps),
				math.Max(initState[1]/factor, eps),
				math.Max(initState[2]/factor, eps),
			},
			MaxState: []float64{
				initState[0] * factor,
				initState[1] * factor,
				initState[2] * factor,
			},
		},
	}
}

// sortedModelVariantKeys returns map keys sorted by ModelID then GPUType then TPDegree (R2: determinism).
func sortedModelVariantKeys[V any](groups map[modelVariantKey]V) []modelVariantKey {
	keys := make([]modelVariantKey, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].ModelID != keys[j].ModelID {
			return keys[i].ModelID < keys[j].ModelID
		}
		if keys[i].Variant.GPUType != keys[j].Variant.GPUType {
			return keys[i].Variant.GPUType < keys[j].Variant.GPUType
		}
		return keys[i].Variant.TPDegree < keys[j].Variant.TPDegree
	})
	return keys
}

// min32 returns the smaller of two float32 values.
func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
