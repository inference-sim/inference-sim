// Command harnessfiles writes, or checks, the files an engine repository keeps as copies of
// blis-substrate's contracts/harness (X7): the golangci-lint settings and the oracle's guard
// scripts. Run it from the engine repository's root, so the copies come from the substrate
// version its go.mod selects:
//
//	go run github.com/inference-sim/blis-substrate/contracts/harness/cmd/harnessfiles          # write
//	go run github.com/inference-sim/blis-substrate/contracts/harness/cmd/harnessfiles -check   # fail if stale
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/inference-sim/blis-substrate/contracts/harness"
)

func main() {
	check := flag.Bool("check", false, "fail, without writing, if a copy is missing or differs")
	root := flag.String("root", ".", "the engine repository's root")
	flag.Parse()
	if *check {
		bad, err := harness.StaleCopies(*root)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		for _, b := range bad {
			fmt.Println(b)
		}
		if len(bad) > 0 {
			os.Exit(1)
		}
		return
	}
	if err := harness.WriteCopies(*root); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
}
