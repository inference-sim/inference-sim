package scripts_test

import (
	"strings"
	"testing"
)

// Flow 3 wiring (#1806): the qa-review answerer and adjudicator must read comment text ONLY through
// deliver-trusted-comments.sh, so a stranger's comment on a public issue/PR never enters their LLM
// prompt. The write-access filtering itself is proven in deliver_trusted_comments_test.go; here we
// pin that the Python tools ROUTE through the filter and that a filter read-failure surfaces as an
// UNREAD marker rather than as an empty comment section (which an agent would read as "no comments").

// tool_gh_issue reads the issue HEAD via gh but its COMMENTS via deliver-trusted-comments.sh, and a
// COMMENT-READ-FAILED from the filter is passed through, never swallowed. Monkeypatches
// subprocess.run so there is no network and no model. Kept identical across both agents (#1792), so
// it is asserted for both.
const commentFilterRoutingProbe = `
import sys, importlib
mod = importlib.import_module(sys.argv[1])
calls = []
class P(object):
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err
def run(argv, **kw):
    calls.append(argv)
    if argv[0] == "gh":
        return P(0, "#1792 Title\n\nBody", "")
    if argv[0] == "bash":
        # The filter degraded: first line COMMENT-READ-FAILED, non-zero exit (as the sh does).
        return P(3, "COMMENT-READ-FAILED\n\nthe channel could not be read\n", "boom")
    raise AssertionError("unexpected subprocess: %r" % (argv,))
mod.subprocess.run = run
res = mod.tool_gh_issue(".", "1792")
# The comment channel is routed through the filter script, not a raw gh comments read.
assert any(a[0] == "bash" and "deliver-trusted-comments.sh" in a[1] for a in calls), calls
# No raw comment read remains beside it.
for a in calls:
    assert "--comments" not in a, a
# The issue HEAD (body) is still present, AND the filter's read-failure is surfaced as UNREAD.
assert "Body" in res, res
assert "COMMENT-READ-FAILED" in res, res
print("OK")
`

func TestQaTools_ReadCommentsThroughTheFilterAndSurfaceUnread(t *testing.T) {
	requirePython3(t)
	for _, mod := range []string{"answerer", "adjudicator"} {
		t.Run(mod, func(t *testing.T) {
			stdout, stderr, code := runPython(t, "", "-c", commentFilterRoutingProbe, mod)
			if code != 0 {
				t.Fatalf("comment-filter routing probe exit=%d stdout=%s stderr=%s", code, stdout, stderr)
			}
			if !strings.Contains(stdout, "OK") {
				t.Fatalf("%s comment-filter routing not confirmed: stdout=%q stderr=%q", mod, stdout, stderr)
			}
		})
	}
}
