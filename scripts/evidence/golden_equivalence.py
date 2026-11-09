"""Golden equivalence: the byte gate's six runs, written as Result schema 1 and as schema 2, compared
figure by figure through the rename table (docs/guide/results.md, "From schema 1 to schema 2").

The claim it checks: the Result v2 goldens (testdata/bytegate/golden) differ from the v1 goldens
in shape only; no figure of the six runs moves. Prints, per case, how many figures were compared
and every value that differs; exits 1 if any differs.

    scripts/evidence/golden_equivalence.py run  V1_BLIS V2_BLIS SUBSTRATE_DIR OUT_DIR
    scripts/evidence/golden_equivalence.py compare OUT_DIR
    scripts/evidence/golden_equivalence.py selftest OUT_DIR    # the non-vacuity check

V1_BLIS is a blis built from commit 1f1a3bb5 (the parent of the Result v2 commit, which writes
schema 1); V2_BLIS is built from the Result v2 commit; SUBSTRATE_DIR is the blis-substrate module
directory the v2 build pins (`go list -m -f '{{.Dir}}' github.com/inference-sim/blis-substrate`),
whose catalog/ and registry/ the v1 build is given, so both read the same data.
"""
import copy, json, os, shlex, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cases():
    out = []
    for line in open(os.path.join(ROOT, 'testdata/bytegate/cases.txt')):
        line = line.strip()
        if line and not line.startswith('#'):
            name, args = line.split(':', 1)
            out.append((name.strip(), shlex.split(args)))
    return out


def run_all(v1, v2, substrate, out):
    os.makedirs(out, exist_ok=True)
    scen = os.path.join(ROOT, 'testdata/bytegate/scenarios')
    for name, args in cases():
        for tag, blis, extra in (('v1', v1, ['--catalog', substrate + '/catalog', '--registry', substrate + '/registry']),
                                 ('v2', v2, [])):
            env = {'GOMAXPROCS': '1', 'PATH': os.environ['PATH'], 'HOME': out}
            r = subprocess.run([blis] + args + ['--scenarios', scen] + extra, capture_output=True, env=env, cwd=out)
            if r.returncode:
                sys.exit(f'{name} {tag}: exit {r.returncode}\n{r.stderr.decode()[-500:]}')
            open(f'{out}/{name}.{tag}.json', 'wb').write(r.stdout)
    print(f'ran {len(cases())} cases into {out}')


def get(doc, path):
    cur = doc
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def compare_docs(a, b):
    """(figures compared, [(label, v1, v2)] differing) for one v1 document and its v2 twin."""
    diffs, n = [], 0

    def cmp(label, x, y):
        nonlocal n
        n += 1
        if x != y:
            diffs.append((label, x, y))

    for k, v in a['metrics'].items():
        k2 = k.replace('itl_', 'tpot_') if k.startswith('itl_') else k
        f = b['metrics'].get(k2)
        cmp(f'metrics.{k}', v, None if f is None else f['value'])
    for k in ['step_interval_mean_ms', 'step_interval_p50_ms', 'step_interval_p90_ms', 'step_interval_p95_ms', 'step_interval_p99_ms']:
        f = b['metrics'].get(k)
        cmp(f'metrics.{k}', get(a, 'engine.cluster.' + k), None if f is None else f['value'])
    if a.get('goodput'):
        for k in ['goodput_rps', 'slo_attainment']:
            f = b['metrics'].get(k)
            cmp(f'metrics.{k}', a['goodput'][k], None if f is None else f['value'])
    q1, q2 = a['requests'], b['requests']
    drops = q2.get('dropped', {})
    cmp('requests.generated', q1['generated'], q2['generated'])
    cmp('requests.completed', q1['completed'], q2['completed'])
    cmp('requests.timed_out', q1['timed_out'], q2['timed_out'])
    cmp('requests.length_capped', q1['length_capped'], q2['length_capped'])
    cmp('requests.in_flight_at_horizon', q1['still_running'], q2['in_flight_at_horizon'])
    cmp('requests.queued_at_horizon', q1['still_queued'] + q1['gateway_queue_depth'], q2['queued_at_horizon'])
    cmp('requests.dropped.admission_rejected', q1['rejected'], drops.get('admission_rejected', 0))
    cmp('requests.dropped.routing_rejected', q1['routing_rejections'] + q1['encode_routing_rejections'],
        drops.get('routing_rejected', 0) + drops.get('encode_routing_rejected', 0))
    cmp('requests.dropped.unservable', q1['dropped_unservable'], drops.get('unservable', 0) + drops.get('decode_kv_unavailable', 0))
    for k in ['gateway_queue_rejected', 'gateway_expired', 'gateway_evicted']:
        cmp(f'requests.dropped.{k}', q1[k], drops.get(k, 0))
    cmp('window.duration_s', get(a, 'window.duration_s'), get(b, 'window.duration_s'))
    cmp('window.requests', get(a, 'window.requests'), get(b, 'window.requests'))
    for sec, new in [('engine', 'reports.engine'), ('router', 'reports.router'), ('kernel', 'reports.kernel')]:
        x, y = a.get(sec), get(b, new)
        if sec == 'engine' and isinstance(x, dict):
            x = {k: v for k, v in x.items() if k != 'frame'}
        cmp(sec, x, y)
    return n, diffs


def compare(d):
    names = sorted({f.split('.')[0] for f in os.listdir(d) if f.endswith('.v1.json')})
    total, bad = 0, 0
    for c in names:
        a = json.load(open(f'{d}/{c}.v1.json'))
        b = json.load(open(f'{d}/{c}.v2.json'))
        n, diffs = compare_docs(a, b)
        total += n
        bad += len(diffs)
        print(f'{c}: {n} compared, {len(diffs)} differ')
        for label, x, y in diffs:
            xs, ys = json.dumps(x, sort_keys=True)[:160], json.dumps(y, sort_keys=True)[:160]
            print(f'   {label}: v1 {xs} | v2 {ys}')
    print('total compared:', total, 'differ:', bad)
    return bad


def selftest(d):
    """Non-vacuity: a 1e-9 change to one figure, and one changed count, must be reported."""
    name = cases()[0][0]
    a = json.load(open(f'{d}/{name}.v1.json'))
    b = json.load(open(f'{d}/{name}.v2.json'))
    b['metrics']['ttft_p99_ms']['value'] += 1e-9
    b['reports']['engine']['cluster']['preemptions'] += 1
    _, diffs = compare_docs(a, b)
    labels = sorted(l for l, _, _ in diffs)
    print('selftest: injected 1e-9 into ttft_p99_ms and +1 preemption ->', labels)
    return 0 if labels == ['engine', 'metrics.ttft_p99_ms'] else 1


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'run' and len(sys.argv) == 6:
        run_all(*sys.argv[2:6])
    elif cmd == 'compare' and len(sys.argv) == 3:
        sys.exit(1 if compare(sys.argv[2]) else 0)
    elif cmd == 'selftest' and len(sys.argv) == 3:
        sys.exit(selftest(sys.argv[2]))
    else:
        sys.exit(__doc__)
