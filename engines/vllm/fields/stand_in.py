# Original to blis-vllm.
"""Evaluate the `valid_when` clauses of fields.json (the reference of the Go evaluator, fields.Exact): is an unset field an exact stand-in for a value?

    python3 -I fields/stand_in.py VLLM_FLASHINFER_ALLREDUCE_BACKEND mnnvl --tp-group-nodes 2 --compute-capability 10.3

The grammar is the one a consumer of the field list reads (blis-substrate contracts/spec/fields): a list of clauses,
any of which may hold, each a conjunction of its keys. Keys: value_in, value_list_has_any,
value_list_has_none, value_not_prefixed, and tp_group_nodes / compute_capability with a comparison
object (eq, gt, gte, lt, lte). An unknown key, or a fact the caller did not give, makes the clause false.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    """Each env field's estimation block, from the committed field list."""
    with open(os.path.join(os.path.dirname(HERE), "fields.json"), encoding="utf-8") as f:
        return {x["name"]: x["estimation"] for x in json.load(f)["fields"] if "valid_when" in x["estimation"]}


def _cap(s):
    major, _, minor = str(s).partition(".")
    return (int(major), int(minor or 0))


def _cmp(actual, test, conv):
    if actual is None:
        return False
    a = conv(actual)
    ops = {"eq": lambda x, y: x == y, "gt": lambda x, y: x > y, "gte": lambda x, y: x >= y,
           "lt": lambda x, y: x < y, "lte": lambda x, y: x <= y}
    return all(op in ops and ops[op](a, conv(v)) for op, v in test.items())


def clause_holds(clause, value, tp_group_nodes=None, compute_capability=None):
    v = str(value).strip().lower()
    tokens = {t.strip() for t in v.split(",") if t.strip()}
    for key, arg in clause.items():
        if key == "value_in":
            ok = v in {str(a).lower() for a in arg}
        elif key == "value_list_has_any":
            ok = bool(tokens & {str(a).lower() for a in arg})
        elif key == "value_list_has_none":
            ok = not (tokens & {str(a).lower() for a in arg})
        elif key == "value_not_prefixed":
            ok = not any(v.startswith(str(a).lower()) for a in arg)
        elif key == "tp_group_nodes":
            ok = _cmp(tp_group_nodes, arg, int)
        elif key == "compute_capability":
            ok = _cmp(compute_capability, arg, _cap)
        else:
            ok = False
        if not ok:
            return False
    return True


def exact(field, value, **facts):
    """True when an unset `field` stands in exactly for `value`; False when no clause holds or the field has none."""
    entry = load().get(field)
    if entry is None:
        return False
    return any(clause_holds(c, value, **facts) for c in entry["valid_when"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("field")
    ap.add_argument("value")
    ap.add_argument("--tp-group-nodes", type=int)
    ap.add_argument("--compute-capability")
    a = ap.parse_args()
    print(exact(a.field, a.value, tp_group_nodes=a.tp_group_nodes, compute_capability=a.compute_capability))
