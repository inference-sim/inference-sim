# Documentation

<p class="lede">The kernel's documentation site is built and published from blis-substrate,
with the rest of its documentation. These pages are blis-vllm's copy of the kernel's section of
that site. blis-vllm keeps the parts that need its code: the generated pages, written by a
generator that prices kernels from vLLM engine blocks, and the Go examples the pages quote.</p>

## Where things live

| Path | Contents |
|---|---|
| blis-substrate's mkdocs.yml and docs/kernel | The site and its kernel section, which blis-substrate builds and publishes. Its own [documentation page](https://github.com/inference-sim/blis-substrate/blob/main/docs/contributing/documentation.md) describes the build and the releases. |
| `kerneltests/docs/` | This copy of the kernel section, one directory per part of the site. |
| `kerneltests/docs/generated/` | Files written by `kerneltests/internal/docgen`. Never edit these by hand. |
| `kerneltests/examples/example_test.go` | The Go the pages describe, run and checked by `go test`. It opens the kernel from a vLLM engine block, which blis-vllm decodes. |
| `kerneltests/examples/docs_examples_test.go` | The test that ties the pages' example output to the examples. |

A page includes a file with the snippets syntax, `--8<-- "path"`, or a marked section of one
with `--8<-- "path:name"`, where the path is relative to the root of the site's repository.

The pages use American spelling. They are written for engineers who have not read the code:
each opens with a paragraph that stands alone, defines a term before it uses it, and points to
the source for detail rather than reproducing it.

## What keeps the pages correct

All of these run in `go test ./...`.

**Generated files.** `go run ./kerneltests/internal/docgen` writes
`kerneltests/docs/generated/`: the step figure, priced by the kernel; the known divergences,
read from the source comments of the kernel go.mod pins; and the pins, read from `go.mod` and
the constants in blis-substrate's `kernel/new.go`. `TestGeneratedDocsAreCurrent` fails while a
committed file differs from what the generator would write. A change to pricing therefore moves
the figure, and the test says to regenerate it.

**Go examples.** The Go the pages describe is sections of
`kerneltests/examples/example_test.go`, each between a start and an end marker comment that
name it. Go runs each example and compares what it prints with its `// Output:` comment.
`TestDocOutputsMatchTheirExamples` checks that the output a page shows is that example's
output.

**Tables tied to the code.** The tables of assumptions and optional coefficients are checked
against the kernel's code by blis-substrate's own tests (`go test ./kernel/` there), because
the names they list are the kernel's.

No test can check that the prose is right. When a change alters behavior, search these pages
for it and update them in the same pull request.
