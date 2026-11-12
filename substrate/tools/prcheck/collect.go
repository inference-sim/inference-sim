package main

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
)

func gitTree(sha string) (string, error) {
	return git("rev-parse", sha+"^{tree}")
}

func ghJSON(v any, args ...string) error {
	out, err := exec.Command("gh", args...).Output()
	if err != nil {
		var stderr string
		if ee, ok := err.(*exec.ExitError); ok {
			stderr = string(ee.Stderr)
		}
		return fmt.Errorf("gh %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr))
	}
	return json.Unmarshal(out, v)
}
