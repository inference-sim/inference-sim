// Package invariants loads and validates the invariant registry: one record for each
// invariant of the BLIS target design, in the format of design §13.1.
//
// The records live in this directory, one YAML file per family (x.yaml holds X1 to X28,
// sub.yaml holds SUB-1 to SUB-20, and so on), and are compiled into the package with
// go:embed, so every repository reads the registry of the substrate version it pins
// (design §13.2). Embedded returns that copy; Load reads a directory of the same layout.
//
// A record has these fields:
//
//	id         the identifier, such as SUB-6. Its prefix names the family file it is in.
//	section    the design section that defines it, such as §5.
//	title      the bold heading the design gives it, if any (X1 to X9 have one).
//	statement  the design's text of the invariant, in full.
//	scope      the repositories that run any of its checks, sorted. It must equal the set
//	           of the checks' repo fields.
//	checks     one entry per check and repository:
//	             repo       the repository that runs it;
//	             check      what it checks, in the design's words;
//	             runs       pull_request, nightly, release, or settings (a repository
//	                        setting, such as branch protection, that a nightly job audits);
//	             status     ci (it runs in CI today) or new (it does not yet);
//	             test       the test, script or workflow job that runs it; required when
//	                        status is ci;
//	             known_bad  the input, in repo, that must make the check fail (X18);
//	                        optional until X18's target milestone.
//	review     null, or a list of review questions, each with question, reviewer, when (the
//	           design's words) and trigger. A trigger names the repositories and the path
//	           globs whose change adds the question to a pull request, with optional
//	           exclude globs; or it is {manual: true}, which a code owner adds by label
//	           (design §13.2).
//	status     ci (every check is ci), partial (some are), new (none is), or review (the
//	           invariant has no mechanical check, only a review question).
//	target     {status, milestone}: the status the invariant must reach, and the milestone
//	           of the current release plan by which it must reach it. The milestone is null
//	           when the status already equals the target.
//	note       optional: what is missing, in the release plan's words.
//
// Path globs use forward slashes. A "*" matches any run of characters other than "/", a
// "**" path segment matches zero or more whole segments, and "?" matches one character
// other than "/".
package invariants
