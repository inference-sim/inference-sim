# Original to blis-sglang.
"""Mutation check of core/sched (S13): each mutant changes one decision of the port; the replay of
SGLang's traces (TestStepsEqualSGLangs) must fail on every one. Usage: python3 -I oracle/mutate_sched.py [index...]"""
import subprocess, shutil, sys, os
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","core","sched")
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
("policy.go","queue[i].MaxNewTokens > queue[j].MaxNewTokens","queue[i].MaxNewTokens < queue[j].MaxNewTokens","lof order"),
("policy.go","if r.C.PrefixLen <= s.cfg.InBatchPrefixCheckThreshold {","if false {","in-batch prefix check"),
("batch.go","r.C.KV.KVAllocatedLen = b.SeqLens[i]\n\t\tr.C.KV.KVCommittedLen = b.SeqLens[i]","r.C.KV.KVAllocatedLen = b.SeqLens[i]","committed len"),
("scheduler.go","if s.chunkedReq.ExtendEnd > s.chunkedReq.C.PrefixLen {\n\t\t\ts.stashChunkedRequest(s.chunkedReq)\n\t\t}","","chunk not stashed"),
]
only=sys.argv[1:]
res=[]
for i,(f,a,b,name) in enumerate(M):
    if only and str(i) not in only: continue
    p=os.path.join(ROOT,f); src=open(p).read()
    if a not in src: res.append((i,name,"NOT APPLIED")); continue
    open(p,"w").write(src.replace(a,b,1))
    try:
        r=subprocess.run(["go","test","./core/sched/","-run","TestStepsEqual"],cwd=ROOT,env={**os.environ,"GOTOOLCHAIN":"go1.27.2"},capture_output=True,text=True,timeout=300)
        out="caught" if r.returncode!=0 else "SURVIVED"
        if "build failed" in r.stdout: out="BUILD FAILED"
    finally:
        open(p,"w").write(src)
    res.append((i,name,out))
for r in res: print(*r)
