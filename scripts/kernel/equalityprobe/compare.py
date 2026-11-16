#!/usr/bin/env python3
"""Compare two equalityprobe outputs record by record (fixture, pool).

    compare.py OLD.jsonl NEW.jsonl

Prints the number of records and, for each field that differs, how many records differ in it,
with one example. A list field that holds the same items in another order is reported as
"(order only)". Exit status 0 always; read the report.
"""
import json,sys,collections
def load(p):
    out={}
    for l in open(p):
        j=json.loads(l); out[(j['fixture'],j['pool'])]=j
    return out
a=load(sys.argv[1]); b=load(sys.argv[2])
PRICE=['step_ns','stages','dp','kv_bytes','fixed','budget','budget_err','verified','other','error']
diffs=collections.Counter(); ex={}
keys=sorted(set(a)|set(b))
for k in keys:
    x=a.get(k); y=b.get(k)
    if x is None or y is None:
        diffs['missing']+=1; ex.setdefault('missing',k); continue
    for f in sorted(set(x)|set(y)):
        if f in ('fixture','pool'): continue
        if x.get(f)!=y.get(f):
            if f=='origins' or f=='prov':
                # compare as multisets
                def ms(v):
                    if isinstance(v,list): return sorted(json.dumps(i,sort_keys=True) for i in v)
                    if isinstance(v,dict): return {kk:ms(vv) for kk,vv in v.items()}
                    return v
                if ms(x.get(f))==ms(y.get(f)):
                    diffs[f+' (order only)']+=1; ex.setdefault(f+' (order only)',k); continue
            if f=='other':
                sub=[s for s in sorted(set(x[f])|set(y[f])) if x[f].get(s)!=y[f].get(s)]
                for s in sub:
                    diffs['other.'+s]+=1; ex.setdefault('other.'+s,(k, str(x[f].get(s))[:300], str(y[f].get(s))[:300]))
                continue
            diffs[f]+=1; ex.setdefault(f,(k, str(x.get(f))[:300], str(y.get(f))[:300]))
print('records',len(keys))
for d,c in diffs.most_common(): print(c,d,'\n   e.g.',ex[d])
