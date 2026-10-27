// Moved from inference-sim/blis-schemas/spec/deployment/modellen.go at 70246460a379f8b1898fbf6c36a3091364118950.

package engineconfig

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// MaxModelLen is the engine's context-length setting, which has three states rather than
// two, so it is a type of its own rather than an int.
//
//   - Zero, the field omitted: unstated. The engine derives the window from the model
//     configuration and uses it as given.
//   - MaxModelLenAuto, written `auto` or `-1`: derive the window the same way, then
//     shrink it to the longest one whose KV cache fits the memory the engine has left
//     (vLLM v0.31.0 vllm/v1/core/kv_cache_utils.py; `--max-model-len auto` and `-1` both
//     parse to -1 in vllm/utils/argparse_utils.py human_readable_int_or_auto). The fit
//     needs the KV budget, so the window that runs is the resolver's to report.
//   - A positive count: that window, which the engine refuses at startup if it exceeds
//     what the model supports or the KV cache can hold.
//
// Omitted and auto are different deployments: on a part that cannot hold the model's
// full window, the first fails to start and the second starts with a shorter one. A plain
// int would have had to spell auto as -1 and let a consumer that never checked for it
// multiply by a negative context; a distinct type makes every consumer that reads the
// field as a number convert it, and so decide what auto means to it.
type MaxModelLen int

// MaxModelLenAuto requests a window fitted to the KV memory.
const MaxModelLenAuto MaxModelLen = -1

// Auto reports whether the window is to be fitted to memory.
func (m MaxModelLen) Auto() bool { return m == MaxModelLenAuto }

// Stated returns the stated window and true, or zero and false when the field is
// omitted or auto.
func (m MaxModelLen) Stated() (int, bool) {
	if m > 0 {
		return int(m), true
	}
	return 0, false
}

// UnmarshalYAML accepts what the engine's flag does (vllm/utils/argparse_utils.py
// human_readable_int_or_auto): `auto` in any case or `-1`; a whole number; or a number
// with a decimal suffix k, m, g, t (powers of 1000, decimals allowed, truncated) or a
// binary suffix K, M, G, T (powers of 1024, whole numbers only). Any other string is an
// error, not a zero: a misspelled `atuo` must not read as an omitted field.
//
// A stated value that comes to zero is an error too, for the same reason: zero is how
// an omitted field reads, and the engine refuses it ("max_model_len must be a positive
// integer", vllm/config/model.py). So is a value too large for an int, which the
// engine's Python integers would carry and refuse later as longer than the model.
func (m *MaxModelLen) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind != yaml.ScalarNode {
		return fmt.Errorf("line %d: max_model_len must be a whole number or auto", n.Line)
	}
	v, err := parseHumanReadableIntOrAuto(n.Value)
	if err == nil && v == 0 {
		err = fmt.Errorf("must be a positive count or auto; omit the field to leave it unstated")
	}
	if err != nil {
		return fmt.Errorf("line %d: max_model_len %q: %v", n.Line, n.Value, err)
	}
	*m = MaxModelLen(v)
	return nil
}

var digitGroups = regexp.MustCompile(`^[+-]?[0-9]+(_[0-9]+)+$`)

var humanReadable = regexp.MustCompile(`^(\d+(?:\.\d+)?)([kKmMgGtT])$`)

func parseHumanReadableIntOrAuto(s string) (int, error) {
	v := strings.TrimSpace(s)
	if v == "-1" || strings.EqualFold(v, "auto") {
		return -1, nil
	}
	if g := humanReadable.FindStringSubmatch(v); g != nil {
		number, suffix := g[1], g[2]
		if mult, ok := map[string]float64{"k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}[suffix]; ok {
			f, err := strconv.ParseFloat(number, 64)
			if err != nil {
				return 0, err
			}
			// Below the largest int on this platform, not int64's, so a 32-bit build
			// refuses what it cannot hold rather than converting it to garbage.
			if v := f * mult; v < float64(math.MaxInt) {
				return int(v), nil
			}
			return 0, errTooLarge
		}
		shift := map[string]uint{"K": 10, "M": 20, "G": 30, "T": 40}[suffix]
		i, err := strconv.Atoi(number)
		if err != nil {
			if strings.Contains(number, ".") {
				return 0, fmt.Errorf("decimals are not allowed with a binary suffix like %s; did you mean %s%s?",
					suffix, number, strings.ToLower(suffix))
			}
			return 0, errTooLarge
		}
		if i > math.MaxInt>>shift {
			return 0, errTooLarge
		}
		return i << shift, nil
	}
	// Python's int() takes digit-group underscores (PEP 515): 1_000, never _1 or 1__0.
	if digitGroups.MatchString(v) {
		v = strings.ReplaceAll(v, "_", "")
	}
	i, err := strconv.Atoi(v)
	if errors.Is(err, strconv.ErrRange) {
		return 0, errTooLarge
	}
	if err != nil {
		return 0, fmt.Errorf("neither a whole number nor auto")
	}
	return i, nil
}

var errTooLarge = errors.New("too large to be a context length")

// MarshalYAML writes auto as `auto`, so a round trip keeps the spelling the engine's
// flag documents.
func (m MaxModelLen) MarshalYAML() (any, error) {
	if m.Auto() {
		return "auto", nil
	}
	return int(m), nil
}
