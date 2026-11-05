package main

import (
	"fmt"
	"html"
	"regexp"
	"strings"
)

// The design marks each invariant as a list item whose first child is
// <span class="inv">ID</span>, followed by an optional <b>title</b>, the statement
// (<span class="stmt">, possibly with a table), an optional table of checks, and the
// enforcement text (<span class="enf">). Trigger outcomes (T-…) have the same form; they are
// checked when their trigger fires and are not registry records (plan §3.2).
var (
	itemStart  = regexp.MustCompile(`<li><span class="inv">([A-Z]+-?[0-9]+)</span>`)
	tags       = regexp.MustCompile(`<[^>]+>`)
	space      = regexp.MustCompile(`\s+`)
	tableBlock = regexp.MustCompile(`(?s)<div class="tbl">.*?</div>`)
	tableRow   = regexp.MustCompile(`(?s)<tr><td>(.*?)</td><td>(.*?)</td>`)
	leadTitle  = regexp.MustCompile(`(?s)^\s*<b>(.*?)</b>`)
)

func text(s string) string {
	return strings.TrimSpace(space.ReplaceAllString(html.UnescapeString(tags.ReplaceAllString(s, "")), " "))
}

// designItem is what the design states about one invariant.
type designItem struct {
	ID, Title, Statement string
}

// parseDesign returns the standing invariants of the design, in the order it defines them.
func parseDesign(doc string) ([]designItem, error) {
	locs := itemStart.FindAllStringSubmatchIndex(doc, -1)
	var out []designItem
	for k, loc := range locs {
		id := doc[loc[2]:loc[3]]
		if strings.HasPrefix(id, "T-") {
			continue
		}
		end := len(doc)
		if k+1 < len(locs) {
			end = locs[k+1][0]
		}
		body := doc[loc[1]:end]
		if i := strings.Index(body, "</li>"); i >= 0 {
			body = body[:i]
		}
		if i := strings.Index(body, `<span class="enf">`); i >= 0 {
			body = body[:i]
		}
		it := designItem{ID: id}
		if m := leadTitle.FindStringSubmatchIndex(body); m != nil {
			it.Title = text(body[m[2]:m[3]])
			body = body[m[1]:]
		}
		var parts []string
		last := 0
		for _, t := range tableBlock.FindAllStringIndex(body, -1) {
			parts = append(parts, text(body[last:t[0]]))
			tbl := body[t[0]:t[1]]
			last = t[1]
			if strings.Contains(tbl, "<th>Check</th>") {
				continue // the check table of X1 to X9; its rows are the record's checks
			}
			var rows []string
			for _, r := range tableRow.FindAllStringSubmatch(tbl, -1) {
				rows = append(rows, text(r[1])+": "+text(r[2]))
			}
			parts = append(parts, strings.Join(rows, "; ")+".")
		}
		parts = append(parts, text(body[last:]))
		var keep []string
		for _, p := range parts {
			if p != "" {
				keep = append(keep, p)
			}
		}
		it.Statement = strings.Join(keep, " ")
		if it.Statement == "" {
			return nil, fmt.Errorf("design: %s has no statement", id)
		}
		out = append(out, it)
	}
	return out, nil
}

// planRow is one row of the release plan's §3.2 status table.
type planRow struct {
	ID, Status, By string
}

var planCell = regexp.MustCompile(`(?s)<td>(.*?)</td>`)

// parsePlan returns the rows of the plan's status table (the table after the heading
// with id p3-2).
func parsePlan(doc string) ([]planRow, error) {
	i := strings.Index(doc, `id="p3-2"`)
	if i < 0 {
		return nil, fmt.Errorf("plan: no §3.2 heading (id p3-2)")
	}
	j := strings.Index(doc[i:], "</table>")
	if j < 0 {
		return nil, fmt.Errorf("plan: §3.2 has no table")
	}
	var out []planRow
	for _, tr := range strings.Split(doc[i:i+j], "<tr>") {
		cells := planCell.FindAllStringSubmatch(tr, -1)
		if len(cells) < 4 {
			continue
		}
		out = append(out, planRow{ID: text(cells[0][1]), Status: text(cells[2][1]), By: text(cells[3][1])})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("plan: §3.2 table has no rows")
	}
	return out, nil
}
