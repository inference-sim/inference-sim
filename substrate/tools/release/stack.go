package main

import (
	"fmt"
	"regexp"
	"sort"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
)

// StackLock describes one review set (design §13.6): the change it makes, the plan
// milestone it ships in, and each repository's branch head. It is a YAML file:
//
//	set: units-as-types
//	milestone: 2              # the release plan milestone the set ships in
//	milestone_release: false  # true for the set that cuts the milestone release
//	repos:
//	  - repo: blis-substrate
//	    branch: m2/units
//	    head: <40-hex commit>
//	    bump: minor             # substrate, simulator and apps: major, minor or patch
//	  - repo: blis-vllm
//	    branch: m2/units
//	    head: <40-hex commit>
//	    upstream: v0.32.0       # engines and the router: the upstream release modelled
//	    fixes: [48]             # issues this release closes, for X27
//
// Stack CI builds the heads together through a go.work it generates and never commits
// (X11). The bot then merges and tags each repository bottom-up, writes each pin once, and
// runs the next layer's checks against the new tags.
type StackLock struct {
	Set       string `yaml:"set"`
	Milestone int    `yaml:"milestone"`
	// MilestoneRelease marks the set that cuts the milestone's release, which reaches the
	// milestone: every invariant targeted at it must then have its target status.
	MilestoneRelease bool        `yaml:"milestone_release,omitempty"`
	Repos            []StackRepo `yaml:"repos"`
}

// StackRepo is one repository's part of a review set.
type StackRepo struct {
	Repo     string `yaml:"repo"`
	Branch   string `yaml:"branch"`
	Head     string `yaml:"head"`
	Bump     string `yaml:"bump,omitempty"`
	Upstream string `yaml:"upstream,omitempty"`
	Fixes    []int  `yaml:"fixes,omitempty"`
	Waivers  []int  `yaml:"waivers,omitempty"`
	Security bool   `yaml:"security,omitempty"`
}

var commitHash = regexp.MustCompile(`^[0-9a-f]{40}$`)

// ParseStackLock reads and validates a stack.lock, and returns its repositories in release
// order: bottom-up by layer (X10), then by name.
func ParseStackLock(src []byte) (*StackLock, error) {
	var s StackLock
	if err := validate.DecodeStrict("stack.lock", src, &s); err != nil {
		return nil, err
	}
	if s.Set == "" {
		return nil, fmt.Errorf("stack.lock: set is empty")
	}
	if s.Milestone < 1 {
		return nil, fmt.Errorf("stack.lock: milestone %d is not a milestone of the release plan", s.Milestone)
	}
	if len(s.Repos) == 0 {
		return nil, fmt.Errorf("stack.lock: no repositories")
	}
	seen := map[string]bool{}
	for i, r := range s.Repos {
		at := fmt.Sprintf("stack.lock: repos[%d] (%s)", i, r.Repo)
		info, ok := Repos[r.Repo]
		switch {
		case !ok:
			return nil, fmt.Errorf("%s: not a BLIS repository", at)
		case seen[r.Repo]:
			return nil, fmt.Errorf("%s: listed twice; a review set has one head per repository", at)
		case r.Branch == "":
			return nil, fmt.Errorf("%s: branch is empty", at)
		case !commitHash.MatchString(r.Head):
			return nil, fmt.Errorf("%s: head %q is not a full commit hash", at, r.Head)
		}
		seen[r.Repo] = true
		switch info.Kind {
		case KindEngine, KindRouter:
			if r.Upstream == "" || r.Bump != "" {
				return nil, fmt.Errorf("%s: an engine or router names its upstream release, and no bump (LIF-5)", at)
			}
		default:
			if r.Upstream != "" {
				return nil, fmt.Errorf("%s: upstream applies only to engines and the router", at)
			}
			if r.Bump != "major" && r.Bump != "minor" && r.Bump != "patch" {
				return nil, fmt.Errorf("%s: bump %q is not major, minor or patch", at, r.Bump)
			}
		}
	}
	sort.SliceStable(s.Repos, func(i, j int) bool {
		a, b := Repos[s.Repos[i].Repo].Layer, Repos[s.Repos[j].Repo].Layer
		if a != b {
			return a < b
		}
		return s.Repos[i].Repo < s.Repos[j].Repo
	})
	return &s, nil
}
