package cmd

import (
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// repoFiles lists the files of the repository at root that belong to it: those git tracks, and
// those it would track (not ignored). A guard that walks the file system instead reads
// whatever else is in the workspace, such as the agent worktrees and plans .gitignore names, so
// its result depends on uncommitted local state and a clean CI run differs from a developer's
// (#2004, X25). Paths are relative to root, slash-separated and sorted.
func repoFiles(t *testing.T, root string) []string {
	t.Helper()
	out, err := exec.Command("git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").Output()
	if err != nil {
		t.Fatalf("git ls-files in %s: %v (the guards read the repository's own files, which git names)", root, err)
	}
	var files []string
	for _, f := range strings.Split(string(out), "\x00") {
		if f == "" {
			continue
		}
		// A tracked file deleted in the working tree is still listed; it is not a file to read.
		if _, err := os.Stat(filepath.Join(root, f)); err != nil {
			continue
		}
		files = append(files, f)
	}
	sort.Strings(files)
	return files
}

// The repository guards read files through repoFiles, so an ignored tree, here the two
// .gitignore names that broke them (#2004), changes nothing, while a tracked or new unignored
// file is seen.
func TestRepoFiles_IgnoredAgentTreesAreNotRepositoryFiles(t *testing.T) {
	root := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("git", append([]string{"-C", root}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	write := func(rel string) {
		t.Helper()
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git("init", "-q")
	write(".gitignore")
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte(".claude/worktrees/\ndocs/superpowers/\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	write("sim/real.go")
	write("docs/live.md")
	git("add", ".")
	write("sim/new_unignored.go")
	write(".claude/worktrees/w1/sim/cluster/deployment.go")
	write("docs/superpowers/plans/old.md")

	got := strings.Join(repoFiles(t, root), " ")
	want := ".gitignore docs/live.md sim/new_unignored.go sim/real.go"
	if got != want {
		t.Errorf("repoFiles = %q, want %q: ignored agent-local trees must not be repository files", got, want)
	}
}
