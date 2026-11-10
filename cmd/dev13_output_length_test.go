package cmd

// DEV-13: a request's sampled output length is written by the workload generator and read only
// by the trace writer, by replay, and by the engine's stop check. Every other part sees what its
// counterpart in the real system sees: the client's maximum token count and the tokens emitted
// so far. The sampled length is sim.Request.OutputTokens (its length and its content, which the
// engine takes as the end-of-sequence position).
//
// This test type-checks every non-test package under sim/ and cmd/ and lists each function that
// reads the field: a selector x.OutputTokens on a sim.Request that is not the target of an
// assignment and not the key of a composite literal. A function in declaredOutputLengthReaders
// is one of the sites the design names. A function in outputLengthReadersToRemove reads it
// against the rule today; each entry names what replaces the read, and the list may only shrink:
// a function that stops reading it must leave the list, and a new reader fails the test.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

const simPkgPath = "github.com/inference-sim/inference-sim/sim"

// declaredOutputLengthReaders are the sites DEV-13 allows, keyed "dir:Func" (a method as
// Type.Method), each with the role that makes the read lawful.
var declaredOutputLengthReaders = map[string]string{
	"sim:EngineDriver.engineRequest":                     "hands the sampled output to the engine, whose stop check reads it as the end-of-sequence position",
	"sim/cluster:ClusterSimulator.dispatchDisaggregated": "the P/D proxy's prefill leg: the engine's stop check at max_tokens 1 samples the first token of the assigned output",
	"sim/cluster:KVTransferStartedEvent.Execute":         "hands the sampled output to the decode engine's stop check",
	"sim/workload:RequestsToTraceRecords":                "the trace writer",
	"sim/workload:ReExportClosedLoopRecords":             "the trace writer",
}

// outputLengthReadersToRemove read the sampled output where the rule gives them only the tokens
// emitted. Each is debt with the reason it remains; the list may only shrink.
var outputLengthReadersToRemove = map[string]string{
	"sim:NewRequestMetrics":                  "sets NumDecodeTokens, the per-model, per-tenant and per-adapter token totals, from the sampled length; the totals move to the tokens emitted in pull request 2, a validation change",
	"sim:EngineDriver.complete":              "decides the post-decode overhead from the presence of a sampled output; it reads the tokens emitted in pull request 2",
	"sim/workload:SessionManager.OnComplete": "appends the content of the tokens a round emitted to the session's context; it reads the count from the token record (GeneratedTokens) but takes the content, and a corruption check against the sampled length, from OutputTokens, because the engine contract returns counts and no token content (the M2 token record, SIM-12)",
}

func TestDEV13_SampledOutputLengthIsReadOnlyAtDeclaredSites(t *testing.T) {
	var undeclared, stale []string
	seen := map[string]bool{}
	for _, r := range outputLengthReaders(t) {
		seen[r] = true
		_, declared := declaredOutputLengthReaders[r]
		_, debt := outputLengthReadersToRemove[r]
		if !declared && !debt {
			undeclared = append(undeclared, r)
		}
	}
	for _, m := range []map[string]string{declaredOutputLengthReaders, outputLengthReadersToRemove} {
		for k := range m {
			if !seen[k] {
				stale = append(stale, k)
			}
		}
	}
	sort.Strings(stale)
	if len(undeclared) > 0 {
		t.Errorf("these functions read the sampled output length (sim.Request.OutputTokens) and are not a site DEV-13 declares: %v\n"+
			"read the request's MaxOutputLen and the tokens emitted (GeneratedTokens) instead", undeclared)
	}
	if len(stale) > 0 {
		t.Errorf("these entries no longer read the sampled output length: delete them: %v", stale)
	}
}

// The known-bad input (X18): a routing function that reads the sampled output length must be
// found, and one that writes the field, or reads another type's field of the same name, must not.
func TestDEV13_FindsAReadOfTheSampledOutputLength(t *testing.T) {
	dir := t.TempDir()
	src := func(name, body string) string {
		pkgDir := filepath.Join(dir, name)
		if err := os.MkdirAll(pkgDir, 0o755); err != nil {
			t.Fatal(err)
		}
		text := "package probe\n\nimport \"github.com/inference-sim/inference-sim/sim\"\n\nvar _ sim.Request\n\n" + body + "\n"
		if err := os.WriteFile(filepath.Join(pkgDir, "probe.go"), []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
		return pkgDir
	}
	read := func(name, body string) []string { return outputLengthReadersIn(t, src(name, body), "probe", ".") }

	if got := read("bad_len", `func score(r *sim.Request) int { return len(r.OutputTokens) }`); len(got) != 1 {
		t.Errorf("a read of len(r.OutputTokens): %v, want one finding", got)
	}
	if got := read("bad_val", `func score(r sim.Request) bool { return r.OutputTokens == nil }`); len(got) != 1 {
		t.Errorf("a read through a value: %v, want one finding", got)
	}
	if got := read("ok_write", `func set(r *sim.Request) { r.OutputTokens = nil }`); len(got) != 0 {
		t.Errorf("a write of the field: %v, want none", got)
	}
	if got := read("ok_other", `type other struct{ OutputTokens []int }
func score(o other) int { return len(o.OutputTokens) }`); len(got) != 0 {
		t.Errorf("another type's field of the same name: %v, want none", got)
	}
}

// outputLengthReaders returns "dir:Func" for every function under sim/ and cmd/ that reads
// sim.Request.OutputTokens.
func outputLengthReaders(t *testing.T) []string {
	t.Helper()
	var dirs []string
	_ = filepath.WalkDir("../sim", func(p string, d fs.DirEntry, err error) error {
		if err == nil && d.IsDir() && d.Name() != "testdata" {
			dirs = append(dirs, p)
		}
		return nil
	})
	dirs = append(dirs, ".")
	var out []string
	for _, dir := range dirs {
		rel := strings.TrimPrefix(filepath.ToSlash(dir), "../")
		if rel == "." {
			rel = "cmd"
		}
		for _, fn := range outputLengthReadersIn(t, dir, "x/"+rel, dir) {
			out = append(out, rel+":"+fn)
		}
	}
	sort.Strings(out)
	return out
}

// outputLengthReadersIn type-checks the non-test files of dir (dependencies from the build of
// depsDir) and returns the name of each function that reads sim.Request.OutputTokens.
func outputLengthReadersIn(t *testing.T, dir, path, depsDir string) []string {
	t.Helper()
	files, _ := filepath.Glob(filepath.Join(dir, "*.go"))
	fset := token.NewFileSet()
	var parsed []*ast.File
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		af, err := parser.ParseFile(fset, f, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		parsed = append(parsed, af)
	}
	if len(parsed) == 0 {
		return nil
	}
	info := &types.Info{Selections: map[*ast.SelectorExpr]*types.Selection{}}
	if _, err := (&types.Config{Importer: exportImporter(t, fset, depsDir)}).Check(path, fset, parsed, info); err != nil {
		t.Fatalf("type-checking %s: %v", dir, err)
	}
	var out []string
	for _, f := range parsed {
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Body == nil || !readsOutputTokens(fd.Body, info) {
				continue
			}
			name := fd.Name.Name
			if fd.Recv != nil && len(fd.Recv.List) == 1 {
				name = recvName(fd.Recv.List[0].Type) + "." + name
			}
			out = append(out, name)
		}
	}
	return out
}

func recvName(e ast.Expr) string {
	switch x := e.(type) {
	case *ast.StarExpr:
		return recvName(x.X)
	case *ast.Ident:
		return x.Name
	case *ast.IndexExpr:
		return recvName(x.X)
	}
	return "?"
}

// readsOutputTokens reports whether body reads sim.Request.OutputTokens.
func readsOutputTokens(body *ast.BlockStmt, info *types.Info) bool {
	written := map[*ast.SelectorExpr]bool{}
	ast.Inspect(body, func(n ast.Node) bool {
		if as, ok := n.(*ast.AssignStmt); ok {
			for _, l := range as.Lhs {
				if s, ok := l.(*ast.SelectorExpr); ok {
					written[s] = true
				}
			}
		}
		return true
	})
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		s, ok := n.(*ast.SelectorExpr)
		if !ok || s.Sel.Name != "OutputTokens" || written[s] {
			return true
		}
		if sel, ok := info.Selections[s]; ok && sel.Kind() == types.FieldVal && isSimRequest(sel.Recv()) {
			found = true
		}
		return true
	})
	return found
}

func isSimRequest(t types.Type) bool {
	for {
		p, ok := t.(*types.Pointer)
		if !ok {
			break
		}
		t = p.Elem()
	}
	n, ok := types.Unalias(t).(*types.Named)
	if !ok || n.Obj().Name() != "Request" || n.Obj().Pkg() == nil {
		return false
	}
	return strings.HasSuffix(n.Obj().Pkg().Path(), "/sim") || n.Obj().Pkg().Path() == simPkgPath
}
