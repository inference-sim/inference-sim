package cmd

// The Result's data for the dashboard's G1 conditions (design 11.3, VLM-3): per selected
// coefficient, the kernel implementations it is valid for, its fit range and whether a priced
// step read it outside that range or it is unidentified; and the engine's resolution of each
// setting beside the host's. The substrate writes both: the first through the selected-entry
// report, to which each kernel that prices steps registers itself once (Kernel.RecordSelected),
// so the Result reads extrapolation when it is built and recording is cumulative across kernels;
// the second through result.ResolutionsOfPool. The engine registers its own pool's kernel when it
// opens the pool over the run's Data (blis-vllm kernelmodel), once per engine instance, so this
// file does not: it only labels each pool's resolutions (engineResolutions, in engine_build.go).

// pooled is the run's opened pool models: the colocated pool, or the prefill and the decode
// pool of a disaggregated run; empty when no kernel was opened.
func pooled(colocated *engineModel, pd *kernelPools) []*engineModel {
	switch {
	case pd != nil:
		return []*engineModel{pd.prefill, pd.decode}
	case colocated != nil:
		return []*engineModel{colocated}
	}
	return nil
}
