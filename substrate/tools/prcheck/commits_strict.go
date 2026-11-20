package main

import (
	"fmt"
	"strings"
)

// StrictCommitPolicy returns the findings of `prcheck commits -strict` for one commit. It is the
// check the integrator runs on an integration head before any other repository pins it, because
// a pin names a commit that is then fixed for good (inference-sim/blis-substrate#335). It adds
// to CommitPolicy, which it does not replace:
//
//   - the commit carries a good signature (Signature is git's %G? letter, G);
//   - its author is one of the maintainers, or the commit carries a Signed-off-by line of one;
//   - it has exactly one Assisted-by line and exactly one Signed-off-by line. A merge commit is
//     held to the same rule, so it is made with `git merge -S --signoff` and the trailer.
//
// X20 keeps these out of the per-pull-request check, which never fails on a signature or a
// sign-off before the closing step after M4; the strict form is opt-in.
func StrictCommitPolicy(c Commit, maintainers []string) []string {
	var bad []string
	add := func(format string, args ...any) {
		bad = append(bad, fmt.Sprintf("%s: ", short(c.Hash))+fmt.Sprintf(format, args...))
	}
	if c.Signature != "G" {
		add("signature status %q; the commit is not signed with a key git trusts (strict)", c.Signature)
	}
	// A merge commit's message ends with git's own "# Conflicts:" comment lines after the
	// trailers, so the trailers are read line by line, not as a final paragraph.
	var signoffs, assisted []string
	for _, l := range strings.Split(strings.ReplaceAll(c.Message, "\r\n", "\n"), "\n") {
		if m := trailerLine.FindStringSubmatch(l); m != nil {
			switch strings.ToLower(m[1]) {
			case "signed-off-by":
				signoffs = append(signoffs, strings.TrimSpace(m[2]))
			case "assisted-by":
				assisted = append(assisted, strings.TrimSpace(m[2]))
			}
		}
	}
	if len(assisted) != 1 {
		add("%d Assisted-by lines; a strict commit has exactly one", len(assisted))
	}
	if len(signoffs) != 1 {
		add("%d Signed-off-by lines; a strict commit has exactly one", len(signoffs))
	}
	if len(maintainers) == 0 {
		add("no maintainer identity given (-maintainer); the author cannot be checked")
		return bad
	}
	isMaintainer := func(identity string) bool {
		for _, m := range maintainers {
			if strings.EqualFold(strings.TrimSpace(identity), strings.TrimSpace(m)) {
				return true
			}
		}
		return false
	}
	byMaintainer := isMaintainer(c.Author)
	for _, s := range signoffs {
		byMaintainer = byMaintainer || isMaintainer(s)
	}
	if !byMaintainer {
		add("author %s is not a maintainer and no maintainer signed off (strict)", c.Author)
	}
	return bad
}
