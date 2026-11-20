package inferencex

import (
	"compress/gzip"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/inference-sim/blis-substrate/ledger"
)

// The legacy partition is inference-sim's tools/validation/inferencex/partition/
// partition.v5.json.gz (seed 20261016), drawn before the ledger existed (plan §3.3). It labels
// each point FIT or HELD. A point was one run at one concurrency of the old inventory
// (build_inventory.py), and its identifier is reproduced here from the dump's own fields:
//
//	run   = "ixdb-" + sha1(github_run_id|run_attempt|config_id|benchmark_type|isl|osl|image|offload_mode)[:16]
//	point = sha1(run|conc)[:16]
//
// with a missing isl, osl or image written as "None", as Python's str(None) writes it. Points
// the old inventory drew from GitHub artifacts after the dump, or from the API, have no record
// in the pinned dump; they are counted, not mapped.

// LegacyPoint returns the legacy point identifier of a row.
func LegacyPoint(r *Row) string {
	none := func(p *int64) string {
		if p == nil {
			return "None"
		}
		return strconv.FormatInt(*p, 10)
	}
	image := "None"
	if r.Image != nil {
		image = *r.Image
	}
	offload := "None"
	if r.OffloadMode != nil {
		offload = *r.OffloadMode
	}
	gh := "None"
	if r.GitHubRunID != nil {
		gh = strconv.FormatInt(*r.GitHubRunID, 10)
	}
	run := "ixdb-" + sha1Hex16(gh, strconv.FormatInt(r.RunAttempt, 10), strconv.FormatInt(r.ConfigID, 10),
		r.BenchmarkType, none(r.ISL), none(r.OSL), image, offload)
	return sha1Hex16(run, strconv.FormatInt(r.Conc, 10))
}

func sha1Hex16(parts ...string) string {
	sum := sha1.Sum([]byte(strings.Join(parts, "|")))
	return hex.EncodeToString(sum[:])[:16]
}

// ReadPartition reads the legacy partition's labels: its "assignment" object, point -> FIT or
// HELD, and its seed.
func ReadPartition(r io.Reader) (map[string]ledger.Side, int64, error) {
	zr, err := gzip.NewReader(r)
	if err != nil {
		return nil, 0, err
	}
	var doc struct {
		Seed       int64             `json:"seed"`
		Assignment map[string]string `json:"assignment"`
	}
	if err := json.NewDecoder(zr).Decode(&doc); err != nil {
		return nil, 0, fmt.Errorf("partition: %w", err)
	}
	out := make(map[string]ledger.Side, len(doc.Assignment))
	for p, s := range doc.Assignment {
		side := ledger.Side(s)
		if !side.Valid() {
			return nil, 0, fmt.Errorf("partition: point %s has label %q, not FIT or HELD", p, s)
		}
		out[p] = side
	}
	if len(out) == 0 {
		return nil, 0, fmt.Errorf("partition: no assignment")
	}
	return out, doc.Seed, nil
}

// PointUnits maps each legacy point with a record in rows to the units its records fell in.
// A point whose every record was left out maps to no unit.
func PointUnits(rows []Row, units []ledger.Unit, labels map[string]ledger.Side) map[string][]string {
	unitOf := map[string]string{}
	for _, u := range units {
		for _, m := range u.Measurements {
			unitOf[m.Record] = u.UnitID
			for _, d := range m.Duplicates {
				unitOf[d] = u.UnitID
			}
			for _, d := range m.Merged {
				unitOf[d.Record] = u.UnitID
			}
		}
	}
	out := map[string][]string{}
	for i := range rows {
		p := LegacyPoint(&rows[i])
		if _, ok := labels[p]; !ok {
			continue
		}
		us := out[p]
		if us == nil {
			us = []string{}
		}
		if u, ok := unitOf[strconv.FormatInt(rows[i].ID, 10)]; ok && !containsString(us, u) {
			us = append(us, u)
		}
		out[p] = us
	}
	return out
}

func containsString(xs []string, x string) bool {
	for _, y := range xs {
		if x == y {
			return true
		}
	}
	return false
}
