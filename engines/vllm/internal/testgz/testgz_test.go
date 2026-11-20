// Original to blis-vllm.

package testgz

import (
	"bytes"
	"compress/gzip"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeGz(t *testing.T, path string, data []byte) {
	t.Helper()
	var buf bytes.Buffer
	w := gzip.NewWriter(&buf)
	if _, err := w.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestReadFileFindsEitherForm(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "plain.json"), []byte("plain"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeGz(t, filepath.Join(dir, "zipped.json.gz"), []byte("zipped"))
	for name, want := range map[string]string{"plain.json": "plain", "zipped.json": "zipped"} {
		got, err := ReadFile(filepath.Join(dir, name))
		if err != nil || string(got) != want {
			t.Errorf("ReadFile(%s) = %q, %v; want %q", name, got, err, want)
		}
		if !Exists(filepath.Join(dir, name)) {
			t.Errorf("Exists(%s) = false", name)
		}
	}
	if _, err := ReadFile(filepath.Join(dir, "absent.json")); !os.IsNotExist(err) {
		t.Errorf("ReadFile of an absent name: %v, want a not-exist error", err)
	}
	if Exists(filepath.Join(dir, "absent.json")) {
		t.Error("Exists of an absent name is true")
	}
}

func TestReadFileRefusesACorruptArchive(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "bad.json.gz"), []byte("not gzip"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadFile(filepath.Join(dir, "bad.json")); err == nil {
		t.Fatal("a corrupt .gz was read without error")
	}
}

func TestGlobListsEachNameOnceWithoutTheSuffix(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"b.trace.json", "both.trace.json", "other.txt"} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	writeGz(t, filepath.Join(dir, "a.trace.json.gz"), []byte("x"))
	writeGz(t, filepath.Join(dir, "both.trace.json.gz"), []byte("x"))
	got, err := Glob(filepath.Join(dir, "*.trace.json"))
	if err != nil {
		t.Fatal(err)
	}
	var want []string
	for _, n := range []string{"a.trace.json", "b.trace.json", "both.trace.json"} {
		want = append(want, filepath.Join(dir, n))
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Glob = %v, want %v", got, want)
	}
}
