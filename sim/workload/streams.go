package workload

// The workload generator's named random substreams (design §8.4, DEV-6). Each draw is named by
// the component that makes it (a client, a session, a prefix group), its purpose and the entity
// it is for (a request, a round), so a request's random inputs depend only on its own name and
// the scenario seed. Request 17 of client 3 draws the same prompt length whatever requests 0 to
// 16 drew, and a session's round 4 the same lengths whatever the server did with rounds 0 to 3.

import (
	"fmt"
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
)

// The purposes a workload entity draws for. Each is its own substream, so a change to one
// distribution does not move the draws of another.
const (
	purposeArrivals         = "arrivals"         // a client's inter-arrival times
	purposeArrivalSchedule  = "arrival-schedule" // a custom arrival sampler's own draws
	purposeInputLength      = "input-length"     // a request's or round's new prompt length
	purposeOutputLength     = "output-length"    // a request's or round's output length
	purposeInputTokens      = "input-tokens"     // a request's or round's new prompt token IDs
	purposeOutputTokens     = "output-tokens"    // a request's or round's output token IDs
	purposeMultimodal       = "multimodal"       // a request's per-modality counts and lengths
	purposeReasonRatio      = "reason-ratio"     // a round's reasoning share of its output
	purposeThinkTime        = "think-time"       // a follow-up round's think time
	purposeDeadlineOffset   = "deadline-offset"  // a follow-up round's recorded deadline offset
	purposeInputReset       = "input-reset"      // a follow-up round's context-compaction target
	purposeResetTokens      = "reset-tokens"     // a compaction round's re-seeded context tokens
	purposeCacheBuster      = "cache-buster"     // a duplicated session's distinguishing token
	purposePrefixTokens     = "tokens"           // a prefix group's shared token IDs
	purposeWindowChoice     = "window-choice"    // ServeGen's choice of window in a period
	purposeSLOAssignment    = "slo-assignment"   // ServeGen's shuffle of chunks over SLO classes
	purposeCorpusShuffle    = "corpus-shuffle"   // replay's shuffle of a corpus's session order
	componentPrefixGroup    = "workload/prefix-group/"
	componentSession        = "workload/session/"
	componentClient         = "workload/client/"
	componentReplayRequest  = "replay/request"
	componentServeGen       = "workload/servegen"
	componentReplayCorpus   = "replay/corpus"
	componentTraceSynthesis = "workload/trace"
)

// clientStreams is the family of client clientIdx: its position among the spec's clients after
// cohort expansion. Its arrival stream has no entity; each request is a child.
func clientStreams(seed int64, clientIdx int) sim.Streams {
	return sim.Streams{Seed: seed, Component: componentClient + fmt.Sprint(clientIdx)}
}

// requestStreams is the family of the n-th request client clientIdx generates, n counted from 0
// over the client's arrival sequence.
func requestStreams(client sim.Streams, n int64) sim.Streams {
	return client.Child(fmt.Sprintf("request/%d", n))
}

// windowStreams is the family of a time-varying client's k-th lifecycle window, in spec order.
func windowStreams(client sim.Streams, k int) sim.Streams {
	return client.Child(fmt.Sprintf("window/%d", k))
}

// SessionStreams is the family of the session sessionID. Every round of the session draws from
// its own child (roundStreams), whichever code builds the round: the generator for round 0, the
// session manager for each follow-up, or a replay loader.
func SessionStreams(seed int64, sessionID string) sim.Streams {
	return sim.Streams{Seed: seed, Component: componentSession + sessionID}
}

// roundStreams is the family of round r of a session.
func roundStreams(session sim.Streams, r int) sim.Streams {
	return session.Child(fmt.Sprintf("round/%d", r))
}

// prefixStream is the stream of prefix group group's shared tokens: every request of the group,
// generated or replayed, carries the same prefix for the same seed and length.
func prefixStream(seed int64, group string) *rand.Rand {
	return sim.NewStream(seed, sim.StreamName{Component: componentPrefixGroup + group, Purpose: purposePrefixTokens})
}

// replayRecordStreams is the family of a replayed trace record that the session manager does
// not drive: a record of a session is its session's round (roundStreams), any other record is
// named by its request_id. A session record replayed as a plain request so draws the tokens the
// session's own replay would.
func replayRecordStreams(seed int64, rec TraceRecord) sim.Streams {
	if rec.SessionID != "" {
		return roundStreams(SessionStreams(seed, rec.SessionID), rec.RoundIndex)
	}
	return sim.Streams{Seed: seed, Component: componentReplayRequest, Entity: fmt.Sprint(rec.RequestID)}
}

// CorpusShuffleStream is the stream replay's --shuffle-corpus draws the session order from.
func CorpusShuffleStream(seed int64) *rand.Rand {
	return sim.NewStream(seed, sim.StreamName{Component: componentReplayCorpus, Purpose: purposeCorpusShuffle})
}

// tokens draws n token IDs from the family's stream for purpose.
func tokens(s sim.Streams, purpose string, n int) []sim.TokenID {
	return sim.GenerateRandomTokenIDs(s.Stream(purpose), n)
}

// clientSessionID names the n-th session (from 0) client clientIdx starts. The name is the
// session's stream component too, so it is a pure function of the client's position and the
// session's place in the client's sequence.
func clientSessionID(clientIdx, n int) string {
	return fmt.Sprintf("sess_c%d_%d", clientIdx, n)
}

// windowSessionPrefix is the prefix of the session names a time-varying client starts in its
// k-th window: the n-th is the prefix followed by n.
func windowSessionPrefix(clientIdx, k int) string {
	return fmt.Sprintf("sess_c%d_w%d_", clientIdx, k)
}

// requestContent draws one single-shot request's prompt and output from the request's family:
// each length and each token sequence from its own stream. A multimodal client's prompt is its
// modalities' tokens (GenerateMultimodalTokens).
func requestContent(rs sim.Streams, mm *MultimodalSpec, inputSampler, outputSampler LengthSampler) (
	inputTokens, outputTokens []sim.TokenID, textCount, imageCount, audioCount, videoCount int, err error) {
	if mm != nil {
		inputTokens, textCount, imageCount, audioCount, videoCount, err = GenerateMultimodalTokens(rs, mm)
		if err != nil {
			return nil, nil, 0, 0, 0, 0, err
		}
	} else {
		inputTokens = tokens(rs, purposeInputTokens, inputSampler.Sample(rs.Stream(purposeInputLength)))
	}
	outputTokens = tokens(rs, purposeOutputTokens, outputSampler.Sample(rs.Stream(purposeOutputLength)))
	return inputTokens, outputTokens, textCount, imageCount, audioCount, videoCount, nil
}
