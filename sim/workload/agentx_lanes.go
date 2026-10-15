package workload

import "math"

// This file ports aiperf 0.12.0's TrajectorySource (timing/trajectory_source.py): the C
// session-tree lanes, each a sequential-sampler root snapshotted at a seeded instant t*,
// and the live stream states replay resumes from.

// agxState is one live stream of a lane at t* (ConversationState).
type agxState struct {
	conv     int32
	next     int32   // first turn at or after t*
	offsetMs float64 // max(0, ts[next] - t*)
	waiting  bool    // a root whose next turn is the join of a branch with live children
	joinTurn int32   // gated root turn this child feeds, or the waiting root's next turn; -1 none
	branch   int32   // branch a child belongs to; -1 for the root
}

// agxLane is one Trajectory with its snapshot.
type agxLane struct {
	trace   *AgentXTrace
	tStarMs float64
	states  []agxState
	// completed[conv] = turns of conv completed before t* (replay_resume_boundaries).
	completed map[int32]int32
}

// agxSampler is SequentialSampler over the root pool: pool[0], pool[1], ... wrapping.
type agxSampler struct {
	pool []*AgentXTrace
	next int
}

func (s *agxSampler) nextTrace() *AgentXTrace {
	if s.next >= len(s.pool) {
		s.next = 0
	}
	t := s.pool[s.next]
	s.next++
	return t
}

// nextRecycle is next_recycle_conversation_id: the next root with any turns, at most one
// pass over the pool.
func (s *agxSampler) nextRecycle() *AgentXTrace {
	for i := 0; i < max(1, len(s.pool)); i++ {
		if t := s.nextTrace(); len(t.convs[0].turns) > 0 {
			return t
		}
	}
	return nil
}

func agxNextAtOrAfter(c *agxConv, tStar float64) int32 {
	for i := range c.turns {
		if c.turns[i].tsMs >= tStar {
			return int32(i)
		}
	}
	return -1
}

func agxOffset(ts, tStar float64) float64 { return math.Max(0, ts-tStar) }

// agxBuildLanes is TrajectorySource._build_trajectories: lane i takes the sampler's next
// root (skipping unspawnable ones, which still consume the sampler) and snapshots it at
// t* drawn from default_rng(sha256("{seed}:{trace}:{lane}")[:8]).uniform over the window.
func agxBuildLanes(s *agxSampler, concurrency int, seed int64, minRatio, maxRatio float64) []*agxLane {
	var lanes []*agxLane
	maxAttempts := concurrency + 2*max(len(s.pool), 1)
	for attempts := 0; len(lanes) < concurrency && attempts < maxAttempts; attempts++ {
		tr := s.nextTrace()
		if len(tr.convs[0].turns) == 0 {
			continue
		}
		start, end := tr.minMs, tr.maxMs
		dur := end - start
		lo := start + float64(minRatio*dur)
		hi := start + float64(maxRatio*dur)
		if hi < lo {
			hi = lo
		}
		tStar := lo
		if hi != lo {
			tStar = newNumpyGenerator(agentxLaneSeed(seed, tr.ID, len(lanes))).uniform(lo, hi)
		}
		if l := agxSnapshot(tr, tStar); l != nil {
			lanes = append(lanes, l)
		}
	}
	return lanes
}

// agxSnapshot is TrajectorySource._snapshot_for: the streams live at t*.
func agxSnapshot(tr *AgentXTrace, tStar float64) *agxLane {
	root := &tr.convs[0]
	rootNext := agxNextAtOrAfter(root, tStar)
	var children []agxState
	pendingJoin := map[int32]bool{}
	for bi := range tr.branches {
		b := &tr.branches[bi]
		if tStar < b.startMs {
			spawnTS := root.turns[b.spawnTurn].tsMs
			if tStar < spawnTS || (rootNext >= 0 && rootNext <= b.spawnTurn) {
				continue // its spawning turn has not completed at t*
			}
		}
		live := 0
		for _, ci := range b.children {
			cn := agxNextAtOrAfter(&tr.convs[ci], tStar)
			if cn < 0 {
				continue
			}
			children = append(children, agxState{
				conv: ci, next: cn, offsetMs: agxOffset(tr.convs[ci].turns[cn].tsMs, tStar),
				joinTurn: b.joinTurn, branch: int32(bi),
			})
			live++
		}
		if live > 0 && b.joinTurn >= 0 {
			pendingJoin[b.joinTurn] = true
		}
	}
	l := &agxLane{trace: tr, tStarMs: tStar, completed: map[int32]int32{}}
	if rootNext >= 0 {
		w := pendingJoin[rootNext]
		st := agxState{conv: 0, next: rootNext, offsetMs: agxOffset(root.turns[rootNext].tsMs, tStar), joinTurn: -1, branch: -1, waiting: w}
		if w {
			st.joinTurn = rootNext
		}
		l.states = append(l.states, st)
	}
	l.states = append(l.states, children...)
	dispatchable := false
	for _, st := range l.states {
		if !st.waiting {
			dispatchable = true
		}
	}
	if !dispatchable {
		return nil
	}
	for _, ci := range tr.reachable() {
		n := agxNextAtOrAfter(&tr.convs[ci], tStar)
		done := int32(len(tr.convs[ci].turns))
		if n >= 0 {
			done = n
		}
		if done > 0 {
			l.completed[ci] = done
		}
	}
	return l
}
