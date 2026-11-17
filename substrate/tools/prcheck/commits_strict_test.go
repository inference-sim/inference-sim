package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const maintainer = "Ada Person <ada@example.com>"

func strictGood() Commit {
	return Commit{Hash: strings.Repeat("a", 40), Author: maintainer, Signature: "G",
		Message: "Merge a thing\n\nAssisted-by: Claude Code (claude-sonnet-5-5)\nSigned-off-by: " + maintainer + "\n"}
}

// The known-bad commits of inference-sim/blis-substrate#335: each is one way the M3.0 history
// failed, and the strict check names it.
func TestStrictCommitPolicyNamesEachWayAHistoryFailed(t *testing.T) {
	none(t, StrictCommitPolicy(strictGood(), []string{maintainer}))
	cases := []struct {
		name string
		edit func(*Commit)
		want string
	}{
		{"unsigned", func(c *Commit) { c.Signature = "N" }, "not signed"},
		{"signed with a key git cannot check", func(c *Commit) { c.Signature = "E" }, "not signed"},
		{"a placeholder author with no maintainer sign-off", func(c *Commit) {
			c.Author = "x <a@b>"
			c.Message = "wip\n\nAssisted-by: Claude Code\nSigned-off-by: Someone Else <else@example.com>\n"
		}, "not a maintainer"},
		{"no Signed-off-by (a merge made without --signoff)", func(c *Commit) {
			c.Message = "Merge a thing\n\nAssisted-by: Claude Code\n"
		}, "0 Signed-off-by"},
		{"no Assisted-by", func(c *Commit) {
			c.Message = "Merge a thing\n\nSigned-off-by: " + maintainer + "\n"
		}, "0 Assisted-by"},
		{"neither trailer", func(c *Commit) { c.Message = "Merge a thing\n" }, "0 Assisted-by"},
		{"two Assisted-by lines", func(c *Commit) {
			c.Message = "Add\n\nAssisted-by: Claude Code\nAssisted-by: Other Tool\nSigned-off-by: " + maintainer + "\n"
		}, "2 Assisted-by"},
		{"two Signed-off-by lines", func(c *Commit) {
			c.Message = "Add\n\nAssisted-by: Claude Code\nSigned-off-by: " + maintainer + "\nSigned-off-by: " + maintainer + "\n"
		}, "2 Signed-off-by"},
	}
	for _, tc := range cases {
		c := strictGood()
		tc.edit(&c)
		t.Run(tc.name, func(t *testing.T) { has(t, StrictCommitPolicy(c, []string{maintainer}), tc.want) })
	}
	// git's "# Conflicts:" comment lines after the trailers of a merge commit do not hide them.
	m := strictGood()
	m.Message += "\n# Conflicts:\n#\tkernel/new.go\n"
	none(t, StrictCommitPolicy(m, []string{maintainer}))
	// A commit by someone else is accepted when a maintainer signs it off.
	c := strictGood()
	c.Author = "Agent Run <run@example.com>"
	none(t, StrictCommitPolicy(c, []string{maintainer}))
	// Without maintainer identities the check cannot pass a commit by accident.
	has(t, StrictCommitPolicy(strictGood(), nil), "no maintainer identity")
}

// The strict check reads what git says: a signed commit by the maintainer passes; an unsigned
// one by a placeholder author fails with both findings. The repository is a temporary one with
// its own identity and an ssh signing key, so no shared configuration is touched.
func TestCommitsStrictReadsSignaturesFromGit(t *testing.T) {
	dir := t.TempDir()
	key := filepath.Join(dir, "key")
	if out, err := exec.Command("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key).CombinedOutput(); err != nil {
		t.Fatalf("ssh-keygen: %v: %s", err, out)
	}
	pub, err := os.ReadFile(key + ".pub")
	if err != nil {
		t.Fatal(err)
	}
	signers := filepath.Join(dir, "signers")
	if err := os.WriteFile(signers, []byte("ada@example.com "+string(pub)), 0o644); err != nil {
		t.Fatal(err)
	}
	repo := filepath.Join(dir, "repo")
	g := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo,
			"-c", "user.name=Ada Person", "-c", "user.email=ada@example.com",
			"-c", "gpg.format=ssh", "-c", "user.signingkey=" + key, "-c", "gpg.ssh.allowedSignersFile=" + signers,
			"-c", "commit.gpgsign=false"}, args...)...)
		var out bytes.Buffer
		cmd.Stdout, cmd.Stderr = &out, &out
		if err := cmd.Run(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out.String())
		}
		return out.String()
	}
	if out, err := exec.Command("git", "init", "-q", "-b", "main", repo).CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	g("commit", "-q", "--allow-empty", "-m", "base")
	g("commit", "-q", "--allow-empty", "-S", "--signoff", "-m", "good\n\nAssisted-by: Claude Code")
	run1 := func(base string) (string, error) {
		wd, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := os.Chdir(wd); err != nil {
				t.Error(err)
			}
		}()
		if err := os.Chdir(repo); err != nil {
			t.Fatal(err)
		}
		// git reads the signers file through its own -c, so the check is given the same config
		// by the environment.
		t.Setenv("GIT_CONFIG_COUNT", "2")
		t.Setenv("GIT_CONFIG_KEY_0", "gpg.format")
		t.Setenv("GIT_CONFIG_VALUE_0", "ssh")
		t.Setenv("GIT_CONFIG_KEY_1", "gpg.ssh.allowedSignersFile")
		t.Setenv("GIT_CONFIG_VALUE_1", signers)
		var w bytes.Buffer
		err := run([]string{"commits", "-strict", "-maintainer", maintainer, "-base", base, "-head", "HEAD"}, &w)
		return w.String(), err
	}
	if out, err := run1("HEAD~1"); err != nil {
		t.Fatalf("a signed, signed-off commit by the maintainer failed the strict check: %v\n%s", err, out)
	}
	if o, err := exec.Command("git", "-C", repo, "-c", "user.name=x", "-c", "user.email=a@b", "commit", "-q", "--allow-empty", "-m", "wip").CombinedOutput(); err != nil {
		t.Fatalf("git commit: %v: %s", err, o)
	}
	out, err := run1("HEAD~1")
	if err == nil {
		t.Fatalf("an unsigned commit by x <a@b> passed the strict check:\n%s", out)
	}
	for _, want := range []string{"not signed", "not a maintainer", "0 Assisted-by", "0 Signed-off-by"} {
		if !strings.Contains(out, want) {
			t.Errorf("no finding contains %q:\n%s", want, out)
		}
	}
}
