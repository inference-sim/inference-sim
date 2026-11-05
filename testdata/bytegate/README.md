# The byte gate

A fixed set of runs whose standard output must not change in a refactor (design SIM-4, X6,
X13). `internal/bytegate` builds `blis` from the checkout, runs each case in `cases.txt` at
GOMAXPROCS 1 and 4 against the catalog and registry of the blis-substrate module `go.mod`
selects, and compares standard output with `golden/<case>.json` byte for byte. The Result's
top-level `runtime` member, which holds the wall clock and the host, is the one part left out.

`scenarios/` holds fixed copies, at the engine release this build runs, so a scenario edited
elsewhere for its own reasons does not move the gate.

A change that moves a golden file is a behaviour change, not a refactor. Regenerate with

    go test ./internal/bytegate -update

and state the diff, the reason and the price-change report in the pull request.
