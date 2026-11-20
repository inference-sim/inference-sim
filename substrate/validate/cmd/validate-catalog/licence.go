package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// The licence check (X12). Only derived values are committed, with one exception: a vendor
// config.json whose licence is on the allow-list (vocab.RedistributableLicences). A model
// whose licence is not on it commits only the file's hash (model.yaml config.sha256) and its
// source revision: the derived-only path.
//
// A committed config.json is redistributed, so its licence's conditions travel with it.
// Apache-2.0 is the repository's own licence (LICENSE). For every other licence,
// licences/index.yaml names the licence text, and any use policy, that the licence requires
// to accompany the file; each text is committed under licences/ with the URL it was read
// from at a pinned revision and its SHA-256. An entry may also quote the attribution
// sentence the licence requires in a notice file, which NOTICE must then carry verbatim.
//
// The check fails on:
//   - a committed config.json whose licence is not on the allow-list;
//   - a committed config.json under a licence other than Apache-2.0 with no index entry;
//   - an index entry whose licence differs from model.yaml's, whose text file is missing
//     or hashes differently, whose notice sentence is missing from its text or from
//     NOTICE, or that names no model with a committed config.json;
//   - a file under licences/ that no entry names.

const (
	licencesDir = "licences"
	indexFile   = "licences/index.yaml"
	noticeFile  = "NOTICE"
)

// ownLicence is the repository's licence. Its text is LICENSE, at the repository root, so a
// config.json under it needs no index entry.
const ownLicence = "Apache-2.0"

type licenceIndex struct {
	Models []licenceEntry `yaml:"models"`
}

type licenceEntry struct {
	Model   string        `yaml:"model"`
	Licence string        `yaml:"licence"`
	Texts   []licenceText `yaml:"texts"`
	Notice  string        `yaml:"notice"`
	Note    string        `yaml:"note"`
}

type licenceText struct {
	File   string `yaml:"file"`
	Source string `yaml:"source"`
	SHA256 string `yaml:"sha256"`
}

// loadIndex reads licences/index.yaml. A catalog without one has no entries; a model that
// needs an entry then fails below.
func loadIndex(root string) (map[string]licenceEntry, error) {
	raw, err := os.ReadFile(filepath.Join(root, indexFile))
	if errors.Is(err, fs.ErrNotExist) {
		return map[string]licenceEntry{}, nil
	}
	if err != nil {
		return nil, err
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	var ix licenceIndex
	if err := dec.Decode(&ix); err != nil {
		return nil, err
	}
	out := map[string]licenceEntry{}
	for i, e := range ix.Models {
		if e.Model == "" || e.Licence == "" || len(e.Texts) == 0 {
			return nil, fmt.Errorf("models[%d] needs model, licence and at least one text", i)
		}
		if _, dup := out[e.Model]; dup {
			return nil, fmt.Errorf("%s is listed twice", e.Model)
		}
		out[e.Model] = e
	}
	return out, nil
}

// readNotice returns NOTICE with its whitespace runs folded to single spaces, or "" when the
// file does not exist.
func readNotice(root string) (string, error) {
	raw, err := os.ReadFile(filepath.Join(root, noticeFile))
	if errors.Is(err, fs.ErrNotExist) {
		return "", nil
	}
	return foldSpace(string(raw)), err
}

// foldSpace folds every whitespace run to one space, so a sentence a licence wraps across
// lines matches the same sentence on one line.
func foldSpace(s string) string { return strings.Join(strings.Fields(s), " ") }

// textProblems checks one entry's texts and notice against the files under licences/, and
// returns what is wrong. texts caches each file's bytes; used records the files named.
func textProblems(root string, e licenceEntry, id *model.Identity, notice string, texts map[string][]byte, used map[string]bool) []string {
	var bad []string
	var all strings.Builder
	for _, t := range e.Texts {
		switch {
		case t.File == "" || t.File != filepath.Base(t.File) || t.File == filepath.Base(indexFile):
			bad = append(bad, fmt.Sprintf("text %q: name a file directly under %s/", t.File, licencesDir))
			continue
		case !strings.HasPrefix(t.Source, "https://"):
			bad = append(bad, fmt.Sprintf("%s: source %q is not an https URL", t.File, t.Source))
		case !validate.IsSHA256(t.SHA256):
			bad = append(bad, fmt.Sprintf("%s: sha256 %q is not a lowercase hex SHA-256", t.File, t.SHA256))
		}
		// A text read from the model's own repository must be read at its pinned revision.
		if id != nil && id.Source != nil && id.Source.Repo != "" &&
			strings.Contains(t.Source, "/"+id.Source.Repo+"/") && !strings.Contains(t.Source, id.Source.Revision) {
			bad = append(bad, fmt.Sprintf("%s: source %s does not name the pinned revision %s", t.File, t.Source, id.Source.Revision))
		}
		used[t.File] = true
		b, ok := texts[t.File]
		if !ok {
			var err error
			b, err = os.ReadFile(filepath.Join(root, licencesDir, t.File))
			if err != nil {
				bad = append(bad, fmt.Sprintf("%s/%s is missing: commit the licence text read from %s", licencesDir, t.File, t.Source))
				continue
			}
			texts[t.File] = b
		}
		sum := sha256.Sum256(b)
		if got := hex.EncodeToString(sum[:]); got != t.SHA256 {
			bad = append(bad, fmt.Sprintf("%s/%s has sha256 %s, but the index records %s", licencesDir, t.File, got, t.SHA256))
		}
		all.Write(b)
		all.WriteByte('\n')
	}
	if e.Notice != "" {
		want := foldSpace(e.Notice)
		if !strings.Contains(foldSpace(all.String()), want) {
			bad = append(bad, fmt.Sprintf("notice %q is not in its licence text", e.Notice))
		}
		if !strings.Contains(notice, want) {
			bad = append(bad, fmt.Sprintf("notice %q is not in %s; the licence requires it there verbatim", e.Notice, noticeFile))
		}
	}
	return bad
}

func validateLicences(r *reporter, root string, dirs []string, ids map[string]*model.Identity) {
	if len(dirs) == 0 {
		return
	}
	r.section("# models/*: config.json licences (X12)")
	index, err := loadIndex(root)
	if err != nil {
		r.loadFail(indexFile, "%v", err)
		return
	}
	notice, err := readNotice(root)
	if err != nil {
		r.loadFail(noticeFile, "%v", err)
		return
	}
	texts := map[string][]byte{}
	used := map[string]bool{}
	seen := map[string]bool{}
	for _, dir := range dirs {
		name := filepath.Base(dir)
		label := name + " licence"
		seen[name] = true
		id := ids[name]
		if id == nil || id.Licence == nil {
			continue // reported by the model.yaml check
		}
		lic := id.Licence.ID
		_, statErr := os.Stat(filepath.Join(dir, "config.json"))
		committed := statErr == nil
		allowed := vocab.LicenceRedistributable(lic)
		entry, listed := index[name]
		var bad []string
		switch {
		case committed && !allowed:
			bad = append(bad, fmt.Sprintf("config.json is committed under %s, which is not on the allow-list %v; "+
				"take the derived-only path: remove config.json and commit only its hash (config.sha256) and revision (X12)",
				lic, vocab.RedistributableLicences()))
		case committed && lic != ownLicence && !listed:
			bad = append(bad, fmt.Sprintf("config.json is committed under %s, which has no entry in %s; "+
				"commit the licence text under %s/ and list it, with its source URL and sha256", lic, indexFile, licencesDir))
		case listed && !committed:
			bad = append(bad, fmt.Sprintf("%s lists the model, but it commits no config.json; remove the entry", indexFile))
		}
		if listed {
			if entry.Licence != lic {
				bad = append(bad, fmt.Sprintf("model.yaml records %s but %s lists %s", lic, indexFile, entry.Licence))
			}
			bad = append(bad, textProblems(root, entry, id, notice, texts, used)...)
		}
		for _, b := range bad {
			r.loadFail(label, "%s", b)
		}
		if len(bad) > 0 {
			continue
		}
		switch {
		case listed:
			var files []string
			for _, t := range entry.Texts {
				files = append(files, t.File)
			}
			r.pass(fmt.Sprintf("%-46s %s, text %s", label, lic, strings.Join(files, ", ")))
		default:
			r.pass(fmt.Sprintf("%-46s %s", label, lic))
		}
	}
	var stray []string
	for m := range index {
		if !seen[m] {
			stray = append(stray, m)
		}
	}
	sort.Strings(stray)
	for _, m := range stray {
		r.loadFail(indexFile, "lists %s, which is not a model in this catalog", m)
	}
	// Every file under licences/ is the index or a text an entry names, so a text cannot
	// outlive the models that cite it.
	entries, err := os.ReadDir(filepath.Join(root, licencesDir))
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		r.loadFail(licencesDir, "%v", err)
		return
	}
	for _, e := range entries {
		if n := e.Name(); n != filepath.Base(indexFile) && !used[n] {
			r.loadFail(licencesDir+"/"+n, "no entry in %s names this file; remove it or list it", indexFile)
		}
	}
}
