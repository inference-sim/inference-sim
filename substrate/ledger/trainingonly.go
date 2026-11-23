package ledger

import "fmt"

// TrainingOnlySources are the calibration sources of requirements Table 1.4. They are data to
// fit, select and validate on, never to hold out, anchor or report against (D9, T-D2), so a
// rule for one may assign only those three roles and has no legacy side.
var TrainingOnlySources = []string{"vllm-ci-perf", "sglang-nightly", "sglang-traces"}

var trainingRoles = []Role{RoleFit, RoleSelect, RoleValidate}

func (c *Config) checkTrainingOnly() []error {
	var errs []error
	for _, src := range TrainingOnlySources {
		r, ok := c.Rules[src]
		if !ok {
			continue
		}
		if len(r.Legacy) != 0 {
			errs = append(errs, fmt.Errorf("rules.%s.legacy: %s is a training-only source and has no legacy side", src, src))
		}
		for i, p := range r.CutPoints {
			if !contains(trainingRoles, p.Role) {
				errs = append(errs, fmt.Errorf("rules.%s.cut_points[%d].role: %q; a training-only source is %v", src, i, p.Role, trainingRoles))
			}
		}
	}
	return errs
}
