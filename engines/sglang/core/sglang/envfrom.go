// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/environ.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package sglang

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// envIssue is the open issue that models the environment variables the engine reads no value of.
const envIssue = "inference-sim/inference-sim#2460"

// EnvFrom is the environment a pool states in its engine block's env map: the process environment
// SGLang's scheduler settings read, from base, with each variable the map names set as SGLang's
// environ.py parses it (EnvBool: true, 1, yes, y and their negatives; EnvInt; EnvFloat, environ.py:143-205).
//
// A value SGLang cannot parse is not the same case there and here: EnvField.get warns and falls back to
// the variable's default (environ.py:70-89), so a run that exported "x" for an integer ran with the default;
// EnvFrom refuses it by name instead (deliberate: a scenario that states a value the engine would silently
// replace is not the run it describes; the refusals are tested). Python's float() takes decimal text, inf and
// nan but not C99 hexadecimal floats, which Go's ParseFloat reads, so text with x or X is refused as invalid.
//
// A variable the engine reads no value of is refused by name: it is accepted only at the value that is
// its default (SGLANG_ENABLE_JIT_DEEPGEMM true, environ.py:1348, which only chooses DeepGEMM's kernels;
// SGLANG_RADIX_FORCE_MISS false, environ.py:722) or not at all.
func EnvFrom(base Env, raw map[string]any) (Env, error) {
	names := make([]string, 0, len(raw))
	for n := range raw {
		names = append(names, n)
	}
	sort.Strings(names)
	out := base
	for _, name := range names {
		text := envText(raw[name])
		var err error
		switch name {
		case "SGLANG_INIT_NEW_TOKEN_RATIO":
			out.InitNewTokenRatio, err = envFloat(name, text)
		case "SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR":
			out.MinNewTokenRatioFactor, err = envFloat(name, text)
		case "SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS":
			var n int
			n, err = envInt(name, text)
			out.NewTokenRatioDecaySteps = float64(n)
		case "SGLANG_RETRACT_DECODE_STEPS":
			out.RetractDecodeSteps, err = envInt(name, text)
		case "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION":
			out.ClipMaxNewTokens, err = envInt(name, text)
		case "SGLANG_DISABLE_OVERLAP_AFTER_PREFILL":
			out.DisableOverlapAfterPrefill, err = envBool(name, text)
		case "SGLANG_ENABLE_JIT_DEEPGEMM":
			var on bool
			if on, err = envBool(name, text); err == nil && !on {
				err = fmt.Errorf("environment variable %s=%s is not modelled: it turns DeepGEMM's kernels off, which the cost model does not price (%s)", name, text, envIssue)
			}
		case "SGLANG_RADIX_FORCE_MISS":
			var on bool
			if on, err = envBool(name, text); err == nil && on {
				err = fmt.Errorf("environment variable %s=%s is not modelled: it forces every prefix-cache match to miss (mem_cache/common.py:240, schedule_policy.py:375@297e54fd) (inference-sim/inference-sim#2449)", name, text)
			}
		default:
			err = fmt.Errorf("environment variable %s is not modelled (%s)", name, envIssue)
		}
		if err != nil {
			return Env{}, err
		}
	}
	return out, nil
}

// envText is the text a process environment would hold for a value the scenario wrote.
func envText(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case bool:
		return strconv.FormatBool(x)
	default:
		return fmt.Sprint(x)
	}
}

func envBool(name, text string) (bool, error) {
	switch strings.ToLower(text) {
	case "true", "1", "yes", "y":
		return true, nil
	case "false", "0", "no", "n":
		return false, nil
	}
	return false, fmt.Errorf("environment variable %s: %q is not a valid boolean value (environ.py:143-150@297e54fd)", name, text)
}

func envInt(name, text string) (int, error) {
	n, err := strconv.Atoi(text)
	if err != nil {
		return 0, fmt.Errorf("environment variable %s: %q is not a valid integer value (environ.py:153-158@297e54fd)", name, text)
	}
	return n, nil
}

func envFloat(name, text string) (float64, error) {
	if strings.ContainsAny(text, "xX") {
		return 0, fmt.Errorf("environment variable %s: %q is not a valid float value (environ.py:200-205@297e54fd)", name, text)
	}
	f, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return 0, fmt.Errorf("environment variable %s: %q is not a valid float value (environ.py:200-205@297e54fd)", name, text)
	}
	return f, nil
}
