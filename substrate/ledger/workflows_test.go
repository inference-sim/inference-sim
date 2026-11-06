package ledger

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const goodSeedWorkflow = `name: ledger
on:
  workflow_run:
    workflows: [ledger candidate]
    types: [completed]
  workflow_dispatch:
permissions:
  contents: read
jobs:
  gate:
    if: github.ref == 'refs/heads/main' && github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    environment: ledger
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          ref: main
      - uses: ./.github/actions/go-toolchain
      - env:
          LEDGER_SEED: ${{ secrets.LEDGER_SEED }}
        run: |
          sha=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_ID" --jq .head_sha)
          go run ./ledger/cmd/ledger aggregate -candidate "$sha"
`

const goodCandidateWorkflow = `name: ledger candidate
on:
  pull_request:
permissions:
  contents: read
jobs:
  candidate:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
      - run: echo predict
`

// A nightly job in the shape ledger.yml has: it lists unit-reads artifacts and keeps those
// from main of this repository.
const goodNightlyWorkflow = `name: ledger
on:
  schedule:
    - cron: "41 4 * * *"
permissions:
  contents: read
jobs:
  gate-nightly:
    if: github.ref == 'refs/heads/main' && github.event_name == 'schedule'
    runs-on: ubuntu-latest
    environment: ledger
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          ref: main
      - run: |
          gh api --paginate "repos/$GITHUB_REPOSITORY/actions/artifacts" \
            --jq '.artifacts[] | select(.name | startswith("unit-reads-")) | select(.workflow_run.head_branch == "main" and (.workflow_run.head_repository_id | tostring) == env.REPO_ID) | .id'
`

// A gate job that takes a run id and verifies the run before it downloads.
const goodRunIDWorkflow = `name: ledger
on:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  gate:
    if: github.ref == 'refs/heads/main' && github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    environment: ledger
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          ref: main
      - env:
          RUN_ID: ${{ inputs.run_id }}
        run: |
          run=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_ID" --jq '[.name, .conclusion, (.head_repository.id | tostring)] | join("|")')
          [ "$run" = "ledger candidate|success|$REPO_ID" ]
          sha=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_ID" --jq .head_sha)
          gh run download "$RUN_ID"
          go run ./ledger/cmd/ledger aggregate -candidate "$sha"
`

func lint(t *testing.T, files map[string]string) error {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	_, err := LintWorkflows(dir)
	return err
}

// D7's workflow lint accepts the split shape and refuses each way a seed could reach
// pull-request code.
func TestLintWorkflows(t *testing.T) {
	if err := lint(t, map[string]string{"a.yml": goodSeedWorkflow, "b.yml": goodCandidateWorkflow}); err != nil {
		t.Fatalf("the good workflows were refused: %v", err)
	}
	for name, body := range map[string]string{"nightly": goodNightlyWorkflow, "run id": goodRunIDWorkflow} {
		if err := lint(t, map[string]string{"a.yml": body}); err != nil {
			t.Errorf("the good %s workflow was refused: %v", name, err)
		}
	}
	bad := map[string]map[string]string{
		"no seed job":                    {"b.yml": goodCandidateWorkflow},
		"started by a pull request":      {"a.yml": strings.Replace(goodSeedWorkflow, "  workflow_dispatch:\n", "  workflow_dispatch:\n  pull_request:\n", 1)},
		"started by pull_request_target": {"a.yml": strings.Replace(goodSeedWorkflow, "  workflow_dispatch:\n", "  workflow_dispatch:\n  pull_request_target:\n", 1)},
		"no main guard":                  {"a.yml": strings.Replace(goodSeedWorkflow, "github.ref == 'refs/heads/main' && ", "", 1)},
		"checks out the head":            {"a.yml": strings.Replace(goodSeedWorkflow, "ref: main", "ref: ${{ github.event.workflow_run.head_sha }}", 1)},
		"checks out the default ref":     {"a.yml": strings.Replace(goodSeedWorkflow, "        with:\n          ref: main\n", "", 1)},
		"local action first": {"a.yml": strings.Replace(goodSeedWorkflow, "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1\n        with:\n          ref: main\n      - uses: ./.github/actions/go-toolchain\n",
			"      - uses: ./.github/actions/go-toolchain\n      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1\n        with:\n          ref: main\n", 1)},
		"another repository":           {"a.yml": strings.Replace(goodSeedWorkflow, "ref: main", "ref: main\n          repository: someone/fork", 1)},
		"not a seed job":               {"a.yml": strings.Replace(goodSeedWorkflow, "  gate:\n", "  build:\n", 1)},
		"seed outside the environment": {"a.yml": goodSeedWorkflow, "b.yml": strings.Replace(goodCandidateWorkflow, "      - run: echo predict\n", "      - env:\n          S: ${{ secrets.LEDGER_SEED }}\n        run: echo predict\n", 1)},
		"candidate reads a secret":     {"a.yml": goodSeedWorkflow, "b.yml": strings.Replace(goodCandidateWorkflow, "      - run: echo predict\n", "      - env:\n          S: ${{ secrets.OTHER }}\n        run: echo predict\n", 1)},
		"candidate in an environment":  {"a.yml": goodSeedWorkflow, "b.yml": strings.Replace(goodCandidateWorkflow, "    runs-on: ubuntu-latest\n", "    runs-on: ubuntu-latest\n    environment: ledger\n", 1)},
		"candidate can write":          {"a.yml": goodSeedWorkflow, "b.yml": strings.Replace(goodCandidateWorkflow, "    permissions:\n      contents: read\n", "    permissions:\n      contents: write\n", 1)},
	}
	bad["unit-reads from any branch"] = map[string]string{"a.yml": strings.Replace(goodNightlyWorkflow,
		` | select(.workflow_run.head_branch == "main" and (.workflow_run.head_repository_id | tostring) == env.REPO_ID)`, "", 1)}
	bad["unit-reads without the repository"] = map[string]string{"a.yml": strings.Replace(goodNightlyWorkflow,
		` and (.workflow_run.head_repository_id | tostring) == env.REPO_ID`, "", 1)}
	bad["run id downloaded unchecked"] = map[string]string{"a.yml": strings.Replace(goodRunIDWorkflow,
		`          run=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_ID" --jq '[.name, .conclusion, (.head_repository.id | tostring)] | join("|")')
          [ "$run" = "ledger candidate|success|$REPO_ID" ]
`, "", 1)}
	bad["run id checked for name only"] = map[string]string{"a.yml": strings.Replace(goodRunIDWorkflow,
		`[.name, .conclusion, (.head_repository.id | tostring)] | join("|")`, `.name`, 1)}
	bad["predictions read without the run's commit"] = map[string]string{"a.yml": strings.Replace(goodRunIDWorkflow,
		`          sha=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_ID" --jq .head_sha)
`, "", 1)}
	bad["predictions read without -candidate"] = map[string]string{"a.yml": strings.Replace(goodRunIDWorkflow,
		` -candidate "$sha"`, "", 1)}
	bad["aggregate without the run's commit"] = map[string]string{"a.yml": strings.Replace(goodSeedWorkflow,
		`          sha=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_ID" --jq .head_sha)
`, "", 1)}
	for name, files := range bad {
		if err := lint(t, files); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// The repository's own workflows pass the lint.
func TestRepositoryWorkflows(t *testing.T) {
	n, err := LintWorkflows("../.github/workflows")
	if err != nil {
		t.Fatal(err)
	}
	if n < len(SeedJobs) {
		t.Errorf("%d seed-holding jobs, want at least %d", n, len(SeedJobs))
	}
}
