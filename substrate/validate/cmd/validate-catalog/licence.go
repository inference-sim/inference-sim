package main

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// The licence check (X12). Only derived values are committed, with one exception: a vendor
// config.json whose licence is on the allow-list (vocab.RedistributableLicences). A model
// whose licence is not on it commits only the file's hash (model.yaml config.sha256) and its
// source revision.
//
// While a person decides each licence, catalog/licence-exceptions.yaml lists the models whose
// config.json is committed under a licence not on the allow-list, each with that licence and
// the status "pending licence review". The check fails on:
//   - a committed config.json whose licence is neither allowed nor listed;
//   - a listed model whose recorded licence differs from the listed one, whose licence is
//     allowed, that no longer commits its config.json, or that does not exist.

const exceptionsFile = "licence-exceptions.yaml"

type licenceExceptions struct {
	Pending []struct {
		Model   string `yaml:"model"`
		Licence string `yaml:"licence"`
		Status  string `yaml:"status"`
	} `yaml:"pending"`
}

func loadExceptions(root string) (map[string]string, error) {
	raw, err := os.ReadFile(filepath.Join(root, exceptionsFile))
	if errors.Is(err, fs.ErrNotExist) {
		return map[string]string{}, nil
	}
	if err != nil {
		return nil, err
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	var ex licenceExceptions
	if err := dec.Decode(&ex); err != nil {
		return nil, fmt.Errorf("%s: %w", exceptionsFile, err)
	}
	out := map[string]string{}
	for i, e := range ex.Pending {
		switch {
		case e.Model == "" || e.Licence == "":
			return nil, fmt.Errorf("%s: pending[%d] needs model and licence", exceptionsFile, i)
		case e.Status != "pending licence review":
			return nil, fmt.Errorf("%s: %s: status %q; the only status is \"pending licence review\"", exceptionsFile, e.Model, e.Status)
		}
		if _, dup := out[e.Model]; dup {
			return nil, fmt.Errorf("%s: %s is listed twice", exceptionsFile, e.Model)
		}
		out[e.Model] = e.Licence
	}
	return out, nil
}

func validateLicences(r *reporter, root string, dirs []string, ids map[string]*model.Identity) {
	if len(dirs) == 0 {
		return
	}
	r.section("# models/*: config.json licences (X12)")
	exceptions, err := loadExceptions(root)
	if err != nil {
		r.loadFail(exceptionsFile, "%v", err)
		return
	}
	seen := map[string]bool{}
	for _, dir := range dirs {
		name := filepath.Base(dir)
		label := name + " licence"
		id := ids[name]
		if id == nil || id.Licence == nil {
			continue // reported by the model.yaml check
		}
		lic := id.Licence.ID
		_, statErr := os.Stat(filepath.Join(dir, "config.json"))
		committed := statErr == nil
		listed, isListed := exceptions[name]
		seen[name] = isListed
		allowed := vocab.LicenceRedistributable(lic)
		switch {
		case isListed && allowed:
			r.loadFail(label, "%s is on the allow-list; remove the model from %s", lic, exceptionsFile)
		case isListed && listed != lic:
			r.loadFail(label, "model.yaml records %s but %s lists %s", lic, exceptionsFile, listed)
		case isListed && !committed:
			r.loadFail(label, "config.json is no longer committed; remove the model from %s", exceptionsFile)
		case committed && !allowed && !isListed:
			r.loadFail(label, "config.json is committed under %s, which is not on the allow-list %v; commit only its hash (config.sha256) and revision, or list it in %s for review (X12)", lic, vocab.RedistributableLicences(), exceptionsFile)
		case isListed:
			r.pass(fmt.Sprintf("%-46s %s, pending licence review", label, lic))
		default:
			r.pass(fmt.Sprintf("%-46s %s", label, lic))
		}
	}
	var stray []string
	for m := range exceptions {
		if _, ok := seen[m]; !ok {
			stray = append(stray, m)
		}
	}
	sort.Strings(stray)
	for _, m := range stray {
		r.loadFail(exceptionsFile, "lists %s, which is not a model in this catalog", m)
	}
}
