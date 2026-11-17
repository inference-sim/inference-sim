# Original to blis-vllm.
"""Tier-2 oracle for whether vLLM's model server answers the router's render endpoints
(enable_scale_out, inference-sim#2129).

vLLM registers the render, derender and token-in/token-out routes in
register_scale_out_api_routers (vllm/entrypoints/scale_out/factories.py). When neither the
server's dedicated mode nor `--enable-scale-out` is set, it registers a handler that answers each
of those routes with HTTP 403 PermissionDeniedError instead of the real ones. This runner takes
that decision out of the pinned source with `ast`, as oracle/run_swa_layout.py does, and runs the
statements unmodified in a namespace that holds only what they read: the `args` the server was
started with and the supported tasks of its model. It then reports the status a render call gets.

Two things are read from the source, not written here: the field's default (BaseFrontendArgs.enable_scale_out,
vllm/entrypoints/common/cli_args.py) and the two assignments that decide
`dedicated_mode` and `enabled`. The runner fails if the pinned source no longer has that shape, so
a change to it cannot go unnoticed. The 403 status and message are read from the same function.

Usage: python -I oracle/run_render.py <vllm checkout> <scenario.json | dir> [out dir]
"""
import ast
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
FACTORIES = "vllm/entrypoints/scale_out/factories.py"
CLI_ARGS = "vllm/entrypoints/common/cli_args.py"


def _parse(vllm_root: str, rel: str) -> ast.Module:
    with open(os.path.join(vllm_root, rel), encoding="utf-8") as f:
        return ast.parse(f.read(), rel)


def field_default(vllm_root: str) -> bool:
    """The default of the frontend arguments' enable_scale_out, as the dataclass writes it."""
    tree = _parse(vllm_root, CLI_ARGS)
    found = [n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "BaseFrontendArgs"
             for n in c.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "enable_scale_out"]
    if len(found) != 1 or found[0].value is None:
        raise SystemExit(f"{CLI_ARGS}: expected one BaseFrontendArgs.enable_scale_out with a default")
    return bool(ast.literal_eval(found[0].value))


def decision(vllm_root: str):
    """The function and its two statements that decide whether the scale-out routes are enabled."""
    tree = _parse(vllm_root, FACTORIES)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "register_scale_out_api_routers"]
    if len(fns) != 1:
        raise SystemExit(f"{FACTORIES}: expected one register_scale_out_api_routers")
    stmts = []
    for name in ("dedicated_mode", "enabled"):
        hit = [n for n in fns[0].body if isinstance(n, ast.Assign)
               and [getattr(t, "id", "") for t in n.targets] == [name]]
        if len(hit) != 1:
            raise SystemExit(f"{FACTORIES}: expected one `{name} = ...` in register_scale_out_api_routers")
        stmts.append(hit[0])
    guard = [n for n in fns[0].body if isinstance(n, ast.If) and ast.unparse(n.test) == "not enabled"]
    if len(guard) != 1:
        raise SystemExit(f"{FACTORIES}: expected one `if not enabled:` in register_scale_out_api_routers")
    # the status the disabled handler answers with, as written: create_error_response(message, name, HTTPStatus.X)
    calls = [n for n in ast.walk(guard[0]) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "create_error_response"]
    if len(calls) != 1 or len(calls[0].args) != 3:
        raise SystemExit(f"{FACTORIES}: expected one create_error_response(message, type, status) in the disabled handler")
    status = ast.unparse(calls[0].args[2])
    kind = ast.literal_eval(calls[0].args[1])
    msg_assign = [n for n in guard[0].body if isinstance(n, ast.Assign)
                  and [getattr(t, "id", "") for t in n.targets] == ["message"]]
    if len(msg_assign) != 1:
        raise SystemExit(f"{FACTORIES}: expected one `message = ...` in the disabled branch")
    message = ast.literal_eval(msg_assign[0].value)
    return stmts, status, kind, message


def run_one(vllm_root: str, scenario_path: str) -> dict:
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import run_scheduler  # noqa: E402

    with open(scenario_path) as f:
        sc = json.load(f)
    cfg = sc["config"]
    default = field_default(vllm_root)
    stated = cfg.get("enable_scale_out")
    # `args` is the server's parsed arguments: the field holds its default unless the scenario states it.
    args = types.SimpleNamespace(enable_scale_out=default if stated is None else stated)
    stmts, status_expr, kind, message = decision(vllm_root)
    ns = {"args": args, "supported_tasks": tuple(cfg.get("supported_tasks", ["generate"]))}
    mod = ast.Module(body=stmts, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, FACTORIES, "exec"), ns)  # noqa: S102 - vLLM's own statements, run as written
    from http import HTTPStatus  # noqa: E402

    status_name = status_expr.rsplit(".", 1)[-1]
    forbidden = int(getattr(HTTPStatus, status_name))
    enabled = bool(ns["enabled"])
    trace = {
        "config": {"enable_scale_out": stated, "supported_tasks": list(ns["supported_tasks"])},
        "field_default": default,
        "dedicated_mode": bool(ns["dedicated_mode"]),
        "serves_render": enabled,
        "render_status": 200 if enabled else forbidden,
        "render_error_type": None if enabled else kind,
        "render_error_message": None if enabled else message,
    }
    trace["provenance"] = run_scheduler.provenance(vllm_root, scenario_path)
    return trace


def main() -> None:
    vllm_root, scenario_path = sys.argv[1], sys.argv[2]
    if os.path.isdir(scenario_path):
        out_dir = sys.argv[3] if len(sys.argv) > 3 else scenario_path
        if "testdata/golden" in os.path.abspath(out_dir).replace(os.sep, "/") and \
                os.environ.get("BLIS_VLLM_ALLOW_GOLDEN_UPDATE") != "1":
            sys.exit("refusing to write golden traces: set BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1, and add "
                     "a golden-reviewed: <trace>: <reason> trailer for each trace the commit changes")
        for name in sorted(f for f in os.listdir(scenario_path)
                           if f.endswith(".json") and not f.endswith(".trace.json")):
            trace = run_one(vllm_root, os.path.join(scenario_path, name))
            with open(os.path.join(out_dir, name[:-5] + ".trace.json"), "w") as f:
                json.dump(trace, f, indent=1, sort_keys=True)
        return
    print(json.dumps(run_one(vllm_root, scenario_path), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
