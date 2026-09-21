package cmd

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/sirupsen/logrus"

	sim "github.com/inference-sim/inference-sim/sim"
)

// Catalog provenance (#1732, R1 task S5 of tracker #1727).
//
// A results file used to say nothing about WHICH catalog produced it. Now that the
// catalog is an explicitly located, versioned directory (#1731 `--catalog`/`BLIS_CATALOG`
// + the blis-catalog repository), a result must be attributable, and an EXPERIMENT
// (uncommitted edits to a catalog config.json) must be distinguishable from a
// REPRODUCIBLE run. This file captures the catalog path, its git revision, and a dirty
// flag; sim.EmitOutput writes them into the --metrics-path file ONLY, so stdout stays
// byte-identical (INV-6) — provenance legitimately varies with catalog git state.
//
// There was no existing git-provenance pattern in sim/ or cmd/ to reuse, so the capture
// is net-new. It lives in cmd/ rather than sim/ because sim/ is a library that must not
// shell out or terminate, and because the catalog path is a CLI-level input.

// catalogGitTimeout bounds each git invocation. rev-parse and status are local,
// network-free operations, so this is a stop rather than a budget: it only fires if git
// wedges on a pathological repository. On timeout the run records an unknown revision
// and continues — provenance is metadata, and no simulation result depends on it.
const catalogGitTimeout = 10 * time.Second

// captureCatalogProvenance records the catalog root's path, git revision, and dirty
// state (#1732, AC-1). It NEVER fails a run: a catalog that is absent, not a git
// checkout, has no commits, or cannot be consulted degrades to
// "path recorded, revision sim.UnknownCatalogRevision, dirty false" with one stderr
// warning (AC-5, R1 — observable, not silent). Returns nil only when no catalog was
// resolved at all, so the results file omits the block rather than claiming an empty
// catalog path.
func captureCatalogProvenance(catalogRoot string) *sim.CatalogProvenance {
	if catalogRoot == "" {
		return nil
	}
	p := &sim.CatalogProvenance{Path: catalogRoot, Revision: sim.UnknownCatalogRevision}
	revision, dirty, err := catalogGitState(catalogRoot)
	if err != nil {
		logrus.Warnf("catalog provenance: recording %q with an unknown git revision: %v",
			catalogRoot, err)
		return p
	}
	p.Revision = revision
	p.Dirty = dirty
	return p
}

// catalogGitState reports the git commit the catalog directory is checked out at and
// whether the CATALOG SUBTREE has uncommitted content.
//
// The dirty check is deliberately scoped to the catalog directory (pathspec "."), not to
// the whole containing repository. A catalog is commonly a subdirectory of a larger
// checkout (a blis-catalog clone sitting inside a workspace, say), where an unrelated
// edit elsewhere in the repo would otherwise mark every run an experiment and make the
// flag useless. Untracked files count as dirty: an uncommitted new catalog entry makes
// the catalog's content differ from its committed revision, which is precisely the
// reproducibility question the flag answers.
func catalogGitState(dir string) (revision string, dirty bool, err error) {
	revOut, err := runCatalogGit(dir, "rev-parse", "HEAD")
	if err != nil {
		return "", false, err
	}
	revision = strings.TrimSpace(revOut)
	if revision == "" {
		return "", false, fmt.Errorf("git rev-parse HEAD in %q returned no revision", dir)
	}
	statusOut, err := runCatalogGit(dir, "status", "--porcelain", "--", ".")
	if err != nil {
		return "", false, err
	}
	return revision, strings.TrimSpace(statusOut) != "", nil
}

// runCatalogGit runs one read-only git command with dir as its working directory and
// returns its stdout. Stderr is folded into the error so a diagnostic (e.g.
// "not a git repository") reaches the warning rather than the user's terminal.
func runCatalogGit(dir string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), catalogGitTimeout)
	defer cancel()

	full := append([]string{"-C", dir}, args...)
	c := exec.CommandContext(ctx, "git", full...)
	var stdout, stderr bytes.Buffer
	c.Stdout = &stdout
	c.Stderr = &stderr
	// git must never block waiting for input: these commands are local and
	// credential-free, but a misconfigured environment should fail fast, not hang.
	c.Stdin = nil
	c.Env = append(os.Environ(), "GIT_TERMINAL_PROMPT=0")
	if runErr := c.Run(); runErr != nil {
		detail := strings.TrimSpace(stderr.String())
		if detail == "" {
			return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), runErr)
		}
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), runErr, detail)
	}
	return stdout.String(), nil
}

// catalogProvenanceEmitOptions builds the sim.EmitOutput options that add catalog
// provenance to a results file. It is the SINGLE shared helper both `blis run` and
// `blis replay` call at their EmitOutput sites, so neither command can drift into
// recording provenance the other omits (INV-13, R23).
//
// It returns no options when outputFilePath is empty: provenance is file-only, so a
// stdout-only run must not pay for a git subprocess it would then discard.
func catalogProvenanceEmitOptions(outputFilePath, catalogRoot string) []sim.EmitOption {
	if outputFilePath == "" {
		return nil
	}
	return []sim.EmitOption{sim.WithCatalogProvenance(captureCatalogProvenance(catalogRoot))}
}
