// Package invariants loads and validates the invariant registry: one record for each
// invariant of the BLIS target design, in the format of design §13.1.
//
// The records live in this directory, one YAML file per family (x.yaml holds X1 to X28,
// sub.yaml holds SUB-1 to SUB-20, and so on), and are compiled into the package with
// the embed directive, so every repository reads the invariant registry of the substrate version it pins
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
//	             runs       pull_request, nightly or release for an automated check, or
//	                        practice for a maintainer practice (design §4.2): a rule a
//	                        person with write access keeps for a repository as a whole, such
//	                        as its branch protection or who may cut a release. A practice
//	                        names the repository it concerns and states its checklist item
//	                        in check; no program runs or audits it;
//	             status     ci (it runs in CI today) or new (it does not yet). For a practice,
//	                        ci means the practice is in place, as the maintainers' checklist
//	                        last recorded it;
//	             test       the test, script or workflow job that runs it; required when
//	                        status is ci, except for a practice, which has none;
//	             known_bad  the input, in repo, that must make the check fail (X18);
//	                        optional until X18's target milestone; none for a practice.
//	review     null, or a list of review questions, each with question, expertise, when (the
//	           design's words) and applies_when. A review question is guidance for the review
//	           agent (design §13.2); no check waits on it. Expertise says who knows the answer.
//	           applies_when names the repositories and the path globs whose change lists the
//	           question in a pull request's description, with optional exclude globs; or it is
//	           {manual: true}, which a reviewer adds by label.
//	guidance   optional: the part of the rule an agent follows on every change it writes or
//	           reviews (design §4.2), in the design's words, such as how a commit message
//	           names agent help. The generator renders it into AGENTS.md. A rule is automated
//	           when a program decides it, a maintainer practice when a person with write access
//	           keeps it for a repository as a whole, and agent guidance when it shapes how a
//	           change is written or judged and a program cannot decide it; one rule can be
//	           more than one.
//	status     ci (every check is ci), partial (some are), new (none is), or review (the
//	           invariant has no mechanical check, only a review question).
//	target     {status, milestone}: the status the invariant must reach, and the milestone
//	           of the current release plan by which it must reach it. The milestone is null
//	           when the status already equals the target.
//	note       optional: what is missing, in the release plan's words.
//
// Dependency direction holds for all three kinds (design §4.2, X7). An automated check reads
// only its own repository and the repositories it depends on, never another repository's
// settings, branches, issues or workflows. A practice is kept in the repository it concerns.
// Guidance names only what the repository it is rendered into may know.
//
// Path globs use forward slashes. A "*" matches any run of characters other than "/", a
// "**" path segment matches zero or more whole segments, and "?" matches one character
// other than "/".
package invariants
