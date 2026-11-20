package inferencex

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/substrate/ledger"
)

// The feed is InferenceX's API read as a whole, not record by record: every record it serves
// above the pinned dump's last id joins the ledger once (requirements 3A.1). The dump is a
// frozen archive and the feed is the live source (3A.9, design D3): the dump owns the record
// ids up to its last, the feed owns the ids above it, a record is never pulled again from the
// other side, and a run whose records fall on both sides keeps one measured date, the one its
// dump records give it. Nothing the feed serves is committed (X12): a ledger owner pins the
// records it proposes in sources.yaml, as for any supplement.

// FeedRecord is one record the feed served and the ingest can read.
type FeedRecord struct {
	ID     int64
	SHA256 string
	// URL is the feed URL that served the record.
	URL string
	Row Row
}

// FeedRead is what a feed served: the readable records in increasing id order, and the ids
// of the records the ingest cannot read, by the field that failed.
type FeedRead struct {
	Records    []FeedRecord
	Unreadable map[string][]int64
}

// FetchFeed reads every URL and merges what they serve by record id. A record two URLs serve
// with different content is an error: the feed changed under the read. A record the ingest
// cannot read (no run to take its attempt from, for one) is left out and counted, never
// guessed.
func FetchFeed(urls []string, fetch Fetcher) (*FeedRead, error) {
	if len(urls) == 0 {
		return nil, fmt.Errorf("no feed URL")
	}
	type served struct {
		url string
		raw json.RawMessage
		sha string
	}
	byID := map[int64]served{}
	for _, u := range urls {
		recs, err := fetchServed(ledger.Supplement{ID: "feed", URL: u}, fetch)
		if err != nil {
			return nil, err
		}
		for id, raw := range recs {
			h, err := RecordSHA256(raw)
			if err != nil {
				return nil, fmt.Errorf("feed record %d: %w", id, err)
			}
			if prev, dup := byID[id]; dup {
				if prev.sha != h {
					return nil, fmt.Errorf("feed record %d: %s serves content %s, %s serves %s", id, prev.url, prev.sha, u, h)
				}
				continue
			}
			byID[id] = served{url: u, raw: raw, sha: h}
		}
	}
	ids := make([]int64, 0, len(byID))
	for id := range byID {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	out := &FeedRead{Unreadable: map[string][]int64{}}
	for _, id := range ids {
		s := byID[id]
		row, err := RowFromAPI(s.raw)
		if err != nil {
			why, _, _ := strings.Cut(err.Error(), ":")
			out.Unreadable[why] = append(out.Unreadable[why], id)
			continue
		}
		out.Records = append(out.Records, FeedRecord{ID: id, SHA256: s.sha, URL: s.url, Row: row})
	}
	return out, nil
}

// FeedRecords returns the records of a fetched supplement as feed records, in the order of the
// supplement's pins, so a pinned record the feed URLs do not serve still joins the ingest.
func (d *SupplementData) FeedRecords(s ledger.Supplement) []FeedRecord {
	out := make([]FeedRecord, 0, len(d.Rows))
	for i, r := range d.Rows {
		out = append(out, FeedRecord{ID: r.ID, SHA256: s.Records[i].SHA256, URL: s.URL, Row: r})
	}
	return out
}

// MergePinned adds the records the supplements pin to a feed read. A record the feed also
// serves must have the content its pin states; otherwise the source changed it, and a ledger
// owner repins it in a pull request of its own.
func MergePinned(feed *FeedRead, pinned []FeedRecord) error {
	have := map[int64]string{}
	for _, r := range feed.Records {
		have[r.ID] = r.SHA256
	}
	for _, p := range pinned {
		if sha, ok := have[p.ID]; ok {
			if sha != p.SHA256 {
				return fmt.Errorf("record %d: the feed serves content %s, pinned %s; the source changed the record, so a ledger owner repins it in a pull request of its own", p.ID, sha, p.SHA256)
			}
			continue
		}
		feed.Records = append(feed.Records, p)
	}
	sort.Slice(feed.Records, func(i, j int) bool { return feed.Records[i].ID < feed.Records[j].ID })
	return nil
}

// FeedReport is what reconciling the feed with the dump found.
type FeedReport struct {
	// Cutoff is the last record id of the pinned dump: the dump owns the ids up to it.
	Cutoff int64 `yaml:"cutoff"`
	// Served counts the readable records the feed served.
	Served int `yaml:"served"`
	// APIRecords counts those above the cutoff, which the ingest reads from the feed.
	APIRecords int `yaml:"api_records"`
	// OwnedByDump counts those at or below it, which are never pulled again from the feed.
	OwnedByDump int `yaml:"owned_by_dump"`
	// RowDiffers lists the records the dump owns whose content the feed prints differently.
	// The dump's copy is the one the ledger keeps.
	RowDiffers []int64 `yaml:"row_differs_from_dump,omitempty"`
	// Unreadable lists the records the ingest cannot read, by the field that failed.
	Unreadable map[string][]int64 `yaml:"unreadable,omitempty"`
	// StraddlingRuns lists the benchmark runs that have records on both sides of the cutoff.
	StraddlingRuns []int64 `yaml:"straddling_runs,omitempty"`
	// UnitsBySourceRevision counts the units whose measurements are all from the dump, all from
	// the feed, or from both (SourceRevision).
	UnitsBySourceRevision map[string]int `yaml:"units_by_source_revision"`
}

// SourceRevision says which side owns a record: "dump" up to the cutoff, "api" above it.
func SourceRevision(record string, cutoff int64) string {
	id, err := intOf(json.RawMessage(record), "record")
	if err != nil || id > cutoff {
		return "api"
	}
	return "dump"
}

// Reconcile splits a feed by the dump's last id. It returns the rows the feed owns, in
// increasing id order, and the report. A feed record at or below the cutoff is never ingested,
// whether or not the dump holds it.
func Reconcile(dump []Row, feed []FeedRecord) ([]Row, *FeedReport, error) {
	if len(dump) == 0 {
		return nil, nil, fmt.Errorf("no dump rows: the dump's last record id is the cutoff")
	}
	rep := &FeedReport{Cutoff: dump[len(dump)-1].ID, Served: len(feed), UnitsBySourceRevision: map[string]int{}}
	inDump := map[int64]*Row{}
	dumpRuns := map[int64]bool{}
	for i := range dump {
		inDump[dump[i].ID] = &dump[i]
		if run := BenchmarkRun(dump[i].RunURL); run != nil {
			dumpRuns[*run] = true
		}
	}
	var api []Row
	straddle := map[int64]bool{}
	seen := map[int64]bool{}
	for _, f := range feed {
		if seen[f.ID] {
			return nil, nil, fmt.Errorf("feed record %d appears twice", f.ID)
		}
		seen[f.ID] = true
		if f.ID <= rep.Cutoff {
			rep.OwnedByDump++
			if d, ok := inDump[f.ID]; ok {
				dh, _ := d.measured()
				ah, _ := f.Row.measured()
				if dh != ah {
					rep.RowDiffers = append(rep.RowDiffers, f.ID)
				}
			}
			continue
		}
		api = append(api, f.Row)
		if run := BenchmarkRun(f.Row.RunURL); run != nil && dumpRuns[*run] {
			straddle[*run] = true
		}
	}
	sort.Slice(api, func(i, j int) bool { return api[i].ID < api[j].ID })
	rep.APIRecords = len(api)
	for run := range straddle {
		rep.StraddlingRuns = append(rep.StraddlingRuns, run)
	}
	sort.Slice(rep.StraddlingRuns, func(i, j int) bool { return rep.StraddlingRuns[i] < rep.StraddlingRuns[j] })
	return api, rep, nil
}

// IngestWithFeed ingests the dump together with the feed's records above the dump's last id.
// The units are those of IngestWithSupplements, and the report carries the reconciliation.
func IngestWithFeed(dump []Row, feed *FeedRead, revision, input string) ([]ledger.Unit, *Report, error) {
	api, fr, err := Reconcile(dump, feed.Records)
	if err != nil {
		return nil, nil, err
	}
	fr.Unreadable = feed.Unreadable
	if len(fr.Unreadable) == 0 {
		fr.Unreadable = nil
	}
	units, rep, err := IngestWithSupplements(dump, []SupplementRows{{ID: "feed", Rows: api}}, revision, input)
	if err != nil {
		return nil, nil, err
	}
	for _, u := range units {
		sides := map[string]bool{}
		for _, m := range u.Measurements {
			sides[SourceRevision(m.Record, fr.Cutoff)] = true
			for _, d := range m.Duplicates {
				sides[SourceRevision(d, fr.Cutoff)] = true
			}
			for _, g := range m.Merged {
				sides[SourceRevision(g.Record, fr.Cutoff)] = true
			}
		}
		switch {
		case len(sides) > 1:
			fr.UnitsBySourceRevision["both"]++
		case sides["api"]:
			fr.UnitsBySourceRevision["api"]++
		default:
			fr.UnitsBySourceRevision["dump"]++
		}
	}
	rep.Feed = fr
	return units, rep, nil
}

var runAttempt = regexp.MustCompile(`^(https://.*/actions/runs/(\d+)/attempts/(\d+))`)

// ProposeSupplements groups the feed's records above the cutoff that no supplement pins yet
// into supplements, one per upstream run and attempt, for a ledger owner to add to
// sources.yaml. Each is pinned by record id and content hash (D3).
func ProposeSupplements(feed []FeedRecord, cutoff int64, pinned map[int64]bool) []ledger.Supplement {
	byRun := map[string]*ledger.Supplement{}
	var order []string
	for _, f := range feed {
		if f.ID <= cutoff || pinned[f.ID] || f.Row.RunURL == nil {
			continue
		}
		m := runAttempt.FindStringSubmatch(*f.Row.RunURL)
		if m == nil {
			continue
		}
		s := byRun[m[1]]
		if s == nil {
			s = &ledger.Supplement{ID: fmt.Sprintf("run-%s-attempt-%s", m[2], m[3]), Kind: ledger.SupplementKindAPI,
				URL: f.URL, Run: m[1]}
			byRun[m[1]] = s
			order = append(order, m[1])
		}
		s.Records = append(s.Records, ledger.PinnedRecord{ID: f.ID, SHA256: f.SHA256})
	}
	out := make([]ledger.Supplement, 0, len(order))
	for _, k := range order {
		out = append(out, *byRun[k])
	}
	return out
}
