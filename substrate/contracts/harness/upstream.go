package harness

import (
	"fmt"
	"regexp"
)

// Upstream identifies the upstream release an engine repository models: its version, the tag
// that names it and the full commit the tag points at. The pin is exact: one build of an engine
// repository models one release (design §14.2, §14.3).
type Upstream struct {
	Version string
	Tag     string
	Commit  string
}

var (
	upstreamVersion = regexp.MustCompile(`^\d+\.\d+\.\d+([.+-][0-9A-Za-z.+-]+)?$`)
	fullCommit      = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// Problems lists what is wrong with the record: a version that is not a release number, an empty
// tag, or a commit that is not a full 40-hex commit id. An empty list means the record is well
// formed; whether the tag names that commit is checked in the engine's CI, against a checkout.
func (u Upstream) Problems() []string {
	var bad []string
	if !upstreamVersion.MatchString(u.Version) {
		bad = append(bad, fmt.Sprintf("version %q is not a release number such as 1.2.3", u.Version))
	}
	if u.Tag == "" {
		bad = append(bad, "tag is empty")
	}
	if !fullCommit.MatchString(u.Commit) {
		bad = append(bad, fmt.Sprintf("commit %q is not a full 40-hex commit id", u.Commit))
	}
	return bad
}
