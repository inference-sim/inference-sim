"""Little's-law offered-load check on r7 single-turn points -> $IX_ROOT/kvm/little.json (KV_METRICS sec 6)."""
import json,glob,sys,subprocess,collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402
R=str(PX.ROOT)
out=[]
for f in glob.glob(R+'/work-r7/points/*/*/result.json'):
    x=json.load(open(f))
    if x.get('status')!='scored' or x['scenario']!='single_turn': continue
    mr=x['measured_rows'][0]; m={k:v['value'] for k,v in mr['measured'].items()}
    if 'output_tput_per_gpu' not in m or 'mean_e2el_ms' not in m: continue
    out.append((x['point_id'],mr['row_id'],x['model_slug'],x['gpu'],x['resolved']['tp']*(x['resolved'].get('dp') or 1),x['conc'],m,mr['signed_err_pct'].get('median_tpot_ms')))
ids=','.join(str(o[1]) for o in out)
q=f"select id, osl, (metrics->>'mean_output_tokens_actual')::float, (metrics->>'num_gpus')::float from benchmark_results where id in ({ids})"
r={int(l.split('|')[0]):l.split('|')[1:] for l in subprocess.run(['psql','-d',PX.PGDATABASE,'-At','-c',q],capture_output=True,text=True).stdout.splitlines()}
res=[]
for pid,rid,mod,gpu,tp,conc,m,e in out:
    osl=float(r[rid][0]); 
    implied=m['output_tput_per_gpu']*tp/osl*m['mean_e2el_ms']/1000
    res.append({'point_id':pid,'model':mod,'gpu':gpu,'tp':tp,'conc':conc,'implied_conc':implied,'ratio':implied/conc,'tpot_err':e})
json.dump(res,open(R+'/kvm/little.json','w'))
c=collections.Counter()
for x in res:
    c[x['model'], 'implied<0.5conc' if x['ratio']<0.5 else 'ok']+=1
for k,v in sorted(c.items()): print(k,v)
g=[x for x in res if x['model']=='gpt-oss-120b' and x['gpu']=='h100' and (x['tpot_err'] or 0)>100]
print('gpt-oss h100 tpot>+100%:', [(x['conc'],x['tp'],round(x['implied_conc'],1)) for x in g])
