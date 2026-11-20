package measurement

import (
	"fmt"
	"reflect"
	"regexp"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// secretPatterns match credentials and cluster access that a run copied from a manifest
// or a command line must not carry into a committed file. A match is refused rather than
// redacted: the adapter scrubs, and this is the check that it did.
var secretPatterns = []struct {
	what string
	re   *regexp.Regexp
}{
	{"a Hugging Face token", regexp.MustCompile(`\bhf_[A-Za-z0-9]{30,}`)},
	{"an AWS access key id", regexp.MustCompile(`\b(AKIA|ASIA)[0-9A-Z]{16}\b`)},
	{"a GitHub token", regexp.MustCompile(`\b(gh[pousr]|github_pat)_[A-Za-z0-9_]{30,}`)},
	{"a bearer token", regexp.MustCompile(`(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}`)},
	{"a private key", regexp.MustCompile(`-----BEGIN [A-Z ]*PRIVATE KEY-----`)},
	{"a Kubernetes Secret", regexp.MustCompile(`\bkind:\s*Secret\b|\bsecretKeyRef\b`)},
	{"a kubeconfig", regexp.MustCompile(`(?i)\bkubeconfig\b|\.kube/config`)},
	{"a credential environment variable", regexp.MustCompile(
		`\b(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|GOOGLE_APPLICATION_CREDENTIALS|OPENAI_API_KEY)\s*[=:]\s*\S+`)},
}

// scanSecrets walks every string a run carries and refuses one matching a secret pattern.
func scanSecrets(p *validate.Problems, r *MeasuredRun) {
	walkStrings(reflect.ValueOf(r).Elem(), "", func(at, s string) {
		for _, sp := range secretPatterns {
			if sp.re.MatchString(s) {
				p.Field(at, "contains %s; scrub it before the run is written", sp.what)
			}
		}
	})
}

func walkStrings(v reflect.Value, at string, visit func(at, s string)) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			walkStrings(v.Elem(), at, visit)
		}
	case reflect.String:
		visit(at, v.String())
	case reflect.Struct:
		t := v.Type()
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}
			name := strings.Split(f.Tag.Get("yaml"), ",")[0]
			if name == "-" {
				continue
			}
			if name == "" {
				name = strings.ToLower(f.Name)
			}
			walkStrings(v.Field(i), join(at, name), visit)
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			walkStrings(v.Index(i), fmt.Sprintf("%s[%d]", at, i), visit)
		}
	case reflect.Map:
		keys := v.MapKeys()
		sort.Slice(keys, func(i, j int) bool { return fmt.Sprint(keys[i]) < fmt.Sprint(keys[j]) })
		for _, k := range keys {
			kat := join(at, fmt.Sprint(k))
			if k.Kind() == reflect.String {
				visit(kat, k.String())
			}
			walkStrings(v.MapIndex(k), kat, visit)
		}
	}
}

func join(at, name string) string {
	if at == "" {
		return name
	}
	return at + "." + name
}
