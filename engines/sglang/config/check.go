// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/arg_groups/validation_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package config

// CheckServerArgs is the part of check_server_args (validation_hook.py:172+) that reads the
// scheduler's priority settings. SGLang runs it after resolution (engine.py:1165, server_args.py:508),
// not inside resolve_once, so Resolve does not call it; the engine does, on the resolved arguments.
// The other checks of that function are not yet ported (inference-sim/blis-sglang#49).
func CheckServerArgs(a *ServerArgs) error {
	const site = "sglang/srt/arg_groups/validation_hook.py:check_server_args"
	if a.EnablePriorityScheduling {
		if a.SchedulePolicy != "fcfs" && a.SchedulePolicy != "lof" {
			return &UpstreamRefusal{Site: site, Message: "To use priority scheduling, schedule_policy must be 'fcfs' or 'lof'. '" +
				a.SchedulePolicy + "' is not supported."}
		}
	}
	if a.RetractionPolicy == "priority" && !a.EnablePriorityScheduling {
		return &UpstreamRefusal{Site: site, Message: "--retraction-policy priority requires --enable-priority-scheduling"}
	}
	return nil
}
