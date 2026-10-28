// Moved from blis-latency-kernel/internal/docgen/chart.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// A deliberately small SVG chart writer: two kinds of mark (a line and a band between two
// lines), a log-2 horizontal axis and a linear vertical one, and labels placed at the end of
// each line rather than in a legend. The figures take every color from the page's stylesheet
// (docs/stylesheets/extra.css), so they read the same in the light and dark schemes and carry
// no styling of their own.

// panel is one plotting area of a figure.
type panel struct {
	x, y, w, h float64 // the plotting area, in figure coordinates
	xs         []float64
	yMax       float64
	yStep      float64
	yUnit      string // appended to the top tick label, which then serves as the axis title
	title      string // above the plot, at its left edge
	xLabel     string
}

func (p panel) px(v float64) float64 {
	lo, hi := math.Log2(p.xs[0]), math.Log2(p.xs[len(p.xs)-1])
	return p.x + (math.Log2(v)-lo)/(hi-lo)*p.w
}

func (p panel) py(v float64) float64 { return p.y + p.h - v/p.yMax*p.h }

// axes draws the title, the gridlines, the tick labels and the horizontal axis title. The
// vertical axis has no separate title: its unit rides on the top tick label.
func (p panel) axes(b *strings.Builder) {
	fmt.Fprintf(b, `<text class="panel-title" x="%.1f" y="%.1f">%s</text>`+"\n",
		p.x, p.y-44, p.title)
	for v := 0.0; v <= p.yMax+1e-9; v += p.yStep {
		y := p.py(v)
		fmt.Fprintf(b, `<line class="grid" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>`+"\n",
			p.x, y, p.x+p.w, y)
		label := fmt.Sprintf("%g", v)
		if v+p.yStep > p.yMax+1e-9 {
			label += " " + p.yUnit
		}
		fmt.Fprintf(b, `<text class="tick" x="%.1f" y="%.1f" text-anchor="end">%s</text>`+"\n",
			p.x-6, y+4, label)
	}
	for _, v := range p.xs {
		x := p.px(v)
		fmt.Fprintf(b, `<line class="tick-mark" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>`+"\n",
			x, p.y+p.h, x, p.y+p.h+4)
		fmt.Fprintf(b, `<text class="tick" x="%.1f" y="%.1f" text-anchor="middle">%g</text>`+"\n",
			x, p.y+p.h+17, v)
	}
	fmt.Fprintf(b, `<text class="axis-title" x="%.1f" y="%.1f" text-anchor="middle">%s</text>`+"\n",
		p.x+p.w/2, p.y+p.h+38, p.xLabel)
}

// path is the SVG path through the points (xs[i], ys[i]).
func (p panel) path(ys []float64) string {
	var parts []string
	for i, v := range ys {
		cmd := "L"
		if i == 0 {
			cmd = "M"
		}
		parts = append(parts, fmt.Sprintf("%s%.1f,%.1f", cmd, p.px(p.xs[i]), p.py(v)))
	}
	return strings.Join(parts, " ")
}

// line draws one series in the given CSS class.
func (p panel) line(b *strings.Builder, class string, ys []float64) {
	fmt.Fprintf(b, `<path class="series %s" d="%s"/>`+"\n", class, p.path(ys))
	for i, v := range ys {
		fmt.Fprintf(b, `<circle class="point %s" cx="%.1f" cy="%.1f" r="2.2"/>`+"\n",
			class, p.px(p.xs[i]), p.py(v))
	}
}

// band fills the region between two series.
func (p panel) band(b *strings.Builder, class string, lo, hi []float64) {
	var pts []string
	for i, v := range hi {
		pts = append(pts, fmt.Sprintf("%.1f,%.1f", p.px(p.xs[i]), p.py(v)))
	}
	for i := len(lo) - 1; i >= 0; i-- {
		pts = append(pts, fmt.Sprintf("%.1f,%.1f", p.px(p.xs[i]), p.py(lo[i])))
	}
	fmt.Fprintf(b, `<polygon class="%s" points="%s"/>`+"\n", class, strings.Join(pts, " "))
}

// endLabel is a direct label for a series, placed at its last point.
type endLabel struct {
	text, class string
	y           float64 // the data value at the last point
}

// endLabels writes the labels to the right of the panel, each level with its series' last
// point. Labels that would overlap are spread apart symmetrically about where they belong,
// so a crowded group stays centered on its lines rather than drifting below them.
func (p panel) endLabels(b *strings.Builder, labels []endLabel) {
	const gap = 15.0
	sort.Slice(labels, func(i, j int) bool { return p.py(labels[i].y) < p.py(labels[j].y) })
	// A cluster is a run of labels set gap apart, starting at top. Adding a label below the
	// last cluster either starts a new one or, when it would overlap, merges into it; a
	// merge can make it overlap the cluster above, so merging repeats upward.
	type cluster struct{ want []float64 }
	top := func(c cluster) float64 {
		var sum float64
		for i, w := range c.want {
			sum += w - float64(i)*gap
		}
		return sum / float64(len(c.want))
	}
	var cs []cluster
	for _, l := range labels {
		cs = append(cs, cluster{want: []float64{p.py(l.y)}})
		for len(cs) > 1 {
			a, c := cs[len(cs)-2], cs[len(cs)-1]
			if top(a)+float64(len(a.want))*gap <= top(c) {
				break
			}
			cs = append(cs[:len(cs)-2], cluster{want: append(append([]float64{}, a.want...), c.want...)})
		}
	}
	i := 0
	for _, c := range cs {
		t := top(c)
		for j := range c.want {
			l := labels[i]
			fmt.Fprintf(b, `<text class="label %s" x="%.1f" y="%.1f">%s</text>`+"\n",
				l.class, p.x+p.w+8, t+float64(j)*gap+4, l.text)
			i++
		}
	}
}

// niceCeiling rounds v up to a multiple of step.
func niceCeiling(v, step float64) float64 { return math.Ceil(v/step) * step }
