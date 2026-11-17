# Original to blis-sglang.
"""Mutation check of core/sched (S13): each mutant changes one decision of the port; the replay of
SGLang's traces (TestStepsEqualSGLangs) must fail on every one. Usage: python3 -I oracle/mutate_sched.py [index...]"""
import subprocess, shutil, sys, os, re
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","core","sched")
GOVER="go"+re.search(r"^go (\S+)",open(os.path.join(ROOT,"..","..","go.mod")).read(),re.M).group(1)  # the toolchain go.mod names
M=[
("budget.go","float64(totalTokens) < b.remainingTotal()","float64(totalTokens) <= b.remainingTotal()","admission boundary"),
("budget.go","return b.remainingTotal() > 0 && b.remainingCurrent() > 0","return b.remainingTotal() > 0","capacity ignores current"),
("budget.go","b.totalOffset += float64(immediate) + maxNewTokens","b.totalOffset += float64(immediate)","reserve omits decode tokens"),
("budget.go","b.currentOffset += float64(immediate)","b.currentOffset += float64(immediate) + maxNewTokens","current offset charges decode"),
("req.go","func ceilTo(n, m int) int { return (n + m - 1) / m * m }","func ceilTo(n, m int) int { return n / m * m }","ceil to floor"),
("adder.go","ratio := a.newTokenRatio\n\tif r.IgnoreEOS {\n\t\tratio = 1.0\n\t}","ratio := a.newTokenRatio","ignore_eos ratio"),
("adder.go","extend = *chunkTokensLimit / a.pageSize * a.pageSize","extend = *chunkTokensLimit","chunk not page aligned"),
("adder.go","a.remInputTokens -= extend","a.remInputTokens -= extend / 2","input budget charge"),
("adder.go","if a.remChunkTokens != nil {\n\t\t*a.remChunkTokens -= extend\n\t}","","chunk budget not charged"),
("adder.go","maxNew = min(r.MaxNewTokens, a.s.cfg.ClipMaxNewTokens)\n\t}\n\ta.updatePrefillBudget(r.ExtendLen(), maxNew)","maxNew = 0\n\t}\n\ta.updatePrefillBudget(r.ExtendLen(), maxNew)","final chunk decode reservation"),
("batch.go","return ao > bo","return ao < bo","retraction order output"),
("batch.go","return al > bl","return al < bl","retraction order prompt"),
("batch.go","min(1.0, float64(decoded+retractSteps*len(reqs))/float64(maxNew+1))","min(1.0, float64(decoded)/float64(maxNew+1))","retract ratio steps"),
("batch.go","if b.Reqs[i].C.KV.KVCommittedLen%s.cfg.PageSize == 0 {","if b.Reqs[i].C.KV.KVCommittedLen%s.cfg.PageSize == 1 {","new page test"),
("scheduler.go","s.newTokenRatio.current = max(s.newTokenRatio.current-s.cfg.NewTokenRatio.Decay, s.cfg.NewTokenRatio.Min)","s.newTokenRatio.current = s.newTokenRatio.current-s.cfg.NewTokenRatio.Decay","ratio floor"),
("scheduler.go","s.newTokenRatio.current = s.cfg.NewTokenRatio.Init","s.newTokenRatio.current = s.cfg.NewTokenRatio.Min","idle reset"),
("scheduler.go","s.processedTokens += b.ExtendNumTokens","s.processedTokens += b.Size()","processed tokens"),
("scheduler.go","ppBudget := s.cfg.PPMaxMicroBatchSize - runningBS","ppBudget := s.cfg.PPMaxMicroBatchSize","micro-batch seats"),
("scheduler.go","if s.cfg.IsMixedChunk && !running.IsEmpty() {","if false {","mixed chunk off"),
("scheduler.go","if !last.IsEmpty() && last != running {","if false {","last batch never merged"),
("policy.go","return -float64(r.NumMatchedPrefixTokens)","return float64(r.NumMatchedPrefixTokens)","lpm sign"),
("policy.go","return a.WaitQueueEntryStep < b.WaitQueueEntryStep","return a.WaitQueueEntryStep > b.WaitQueueEntryStep","spf tie"),
("policy.go","ratio: -float64(waited) / float64(uncached)","ratio: float64(waited) / float64(uncached)","hrrn sign"),
("policy.go","return a.MaxNewTokens > b.MaxNewTokens","return a.MaxNewTokens < b.MaxNewTokens","lof order"),
("policy.go","if r.C.PrefixLen <= s.cfg.InBatchPrefixCheckThreshold {","if false {","in-batch prefix check"),
("batch.go","r.C.KV.KVAllocatedLen = b.SeqLens[i]\n\t\tr.C.KV.KVCommittedLen = b.SeqLens[i]","r.C.KV.KVAllocatedLen = b.SeqLens[i]","committed len"),
("scheduler.go","if s.chunkedReq.ExtendEnd > s.chunkedReq.C.PrefixLen {\n\t\t\ts.stashChunkedRequest(s.chunkedReq)\n\t\t}","","chunk not stashed"),
("policy.go","\t\t\tif ca != cb {\n\t\t\t\treturn ca > cb","\t\t\tif ca != cb {\n\t\t\t\treturn ca < cb","routing-key count order"),
("policy.go","\tcase ina != inb:\n\t\t\treturn ina","\tcase ina != inb:\n\t\t\treturn inb","routing-key held first"),
("policy.go","if s.prioritySign() > 0 {\n\t\treturn a < b","if s.prioritySign() < 0 {\n\t\treturn a < b","priority direction"),
("policy.go","\t\treturn a.WaitQueueEntryStep < b.WaitQueueEntryStep\n\t})\n}\n\n// shuffleQueue","\t\treturn a.WaitQueueEntryStep > b.WaitQueueEntryStep\n\t})\n}\n\n// shuffleQueue","priority then arrival"),
("policy.go","\t\tqueue[i] = r\n","\t\t_ = r\n","random order ignored"),
("policy.go","refresh := s.waitingPrefixRefreshCalls%waitingPrefixRefreshInterval == 0","refresh := s.waitingPrefixRefreshCalls%waitingPrefixRefreshInterval == 1","waiting prefix refresh phase"),
("policy.go","for i := len(head) - 1; i >= 0; i-- {\n\t\tr := head[i]","for i := 0; i < len(head); i++ {\n\t\tr := head[i]","waiting prefix touch order"),
("policy.go","if s.cfg.EnablePriorityScheduling && a.Priority != b.Priority {","if false {","lof ignores priority"),
("scheduler.go","len(s.waiting)+1 <= *s.cfg.MaxQueuedRequests","len(s.waiting) <= *s.cfg.MaxQueuedRequests","queue limit boundary"),
("scheduler.go","s.priorityBefore(in.Priority, cand.Priority) {","!s.priorityBefore(in.Priority, cand.Priority) {","queue-full victim"),
("scheduler.go","r.Priority = math.MaxInt64","r.Priority = math.MinInt64","default priority"),
("scheduler.go","r.C.OriginInputIDs = []int64{0}","r.C.OriginInputIDs = []int64{0, 0}","aborted request length"),
("scheduler.go","s.cfg.AbortOnPriorityWhenDisabled {","!s.cfg.AbortOnPriorityWhenDisabled {","priority disabled abort"),
("scheduler.go","len(r.C.OriginInputIDs) >= s.cfg.MaxReqInputLen","len(r.C.OriginInputIDs) > s.cfg.MaxReqInputLen","input length boundary"),
("adder.go","if diff.Cmp(big.NewInt(int64(s.cfg.PriorityPreemptionThreshold))) <= 0 {","if diff.Cmp(big.NewInt(int64(s.cfg.PriorityPreemptionThreshold))) < 0 {","preemption threshold"),
("adder.go","\t\tminToRemove -= a.runningRequestTotalTokenOffset(r)\n","","preemption frees nothing"),
("adder.go","return x.WaitQueueEntryStep > y.WaitQueueEntryStep","return x.WaitQueueEntryStep < y.WaitQueueEntryStep","preempt newest first"),
("adder.go","if curRem+freed-st.left*bs <= ignoreEOSReserveTokens*bs {","if curRem+freed-st.left*bs < ignoreEOSReserveTokens*bs {","ignore_eos reserve"),
("adder.go","i = max(len(a.reqStates)-1, 0)","i = len(a.reqStates)","ignore_eos insertion quirk"),
("adder.go","paged := a.ceilPaged(cand) + a.pageSize","paged := a.ceilPaged(cand)","ignore_eos page overhead"),
("adder.go","a.budget.totalOffset -= a.runningRequestTotalTokenOffset(r)","","preempted offset kept"),
("policy.go","\t\t\tcopy(queue, sorted)\n","","dfs order ignored"),
("adder.go","len(a.canRunList) >= *x {","len(a.canRunList) > *x {","prefill max requests"),
("adder.go","\t\t\treturn NoToken\n\t\t}\n\t\tfreed += float64(st.occupied)","\t\t}\n\t\tfreed += float64(st.occupied)","ignore_eos ledger never refuses"),
("overlap.go","(b.Mode == Extend || b.Mode == Mixed)","b.Mode == Extend","mixed batch is not an extend for overlap"),
("overlap.go","isExtend(batch) && isExtend(last)","isExtend(batch)","consecutive prefill needs only the batch"),
("overlap.go","\tif disable {\n\t\ts.popAndProcess()\n\t}","\tif false {\n\t\ts.popAndProcess()\n\t}","disabled overlap still defers the result"),
("overlap.go","} else if batch == nil && s.isFullyIdle() {","} else if false {","overlap idle never resets the ratio"),
("overlap.go","if queued[r] && r.NextOutputFinishesByLength() {","if r.NextOutputFinishesByLength() {","finishing check ignores the queued result"),
("overlap.go","if queued[r] && r.NextOutputFinishesByLength() {","if queued[r] {","finishing check ignores the length"),
("overlap.go","\tif !s.cfg.EnableOverlap || len(s.resultQueue) == 0 {\n\t\treturn nil","\tif true {\n\t\treturn nil","skip-finishing off"),
("req.go","return len(r.C.OutputIDs)+1 >= r.MaxNewTokens","return len(r.C.OutputIDs)+1 > r.MaxNewTokens","next output finishes boundary"),
("scheduler.go","running.FilterBatch(finishing)\n\t\tif len(finishing) > 0 {\n\t\t\trunning.BatchIsFull = false","running.FilterBatch(nil)\n\t\tif len(finishing) > 0 {\n\t\t\trunning.BatchIsFull = false","mixed batch decodes a finishing request"),
("scheduler.go","running.FilterBatch(finishing)\n\t\tif len(finishing) > 0 {\n\t\t\trunning.BatchIsFull = false","running.FilterBatch(finishing)\n\t\tif len(finishing) > 0 {\n\t\t\t_ = 0","mixed batch stays full after a finishing request"),
("scheduler.go","if retract && len(finishing) > 0 {","if false {","retract despite finishing requests"),
("scheduler.go","s.cfg.EnableOverlap && (r.Finished() || r.IsRetracted)","s.cfg.EnableOverlap && r.IsRetracted","finished row decoded again"),
("scheduler.go","s.cfg.EnableOverlap && (r.Finished() || r.IsRetracted)","s.cfg.EnableOverlap && r.Finished()","retracted row decoded again"),
("scheduler.go","(!s.cfg.EnableOverlap || len(s.resultQueue) == 0) &&","true &&","idle ignores the unprocessed result"),
]
only=sys.argv[1:]
res=[]
for i,(f,a,b,name) in enumerate(M):
    if only and str(i) not in only: continue
    p=os.path.join(ROOT,f); src=open(p).read()
    if a not in src: res.append((i,name,"NOT APPLIED")); continue
    open(p,"w").write(src.replace(a,b,1))
    try:
        r=subprocess.run(["go","test","./core/sched/","-run","TestStepsEqual"],cwd=ROOT,env={**os.environ,"GOTOOLCHAIN":GOVER},capture_output=True,text=True,timeout=300)
        out="caught" if r.returncode!=0 else "SURVIVED"
        if "build failed" in r.stdout: out="BUILD FAILED"
    finally:
        open(p,"w").write(src)
    res.append((i,name,out))
for r in res: print(*r)
