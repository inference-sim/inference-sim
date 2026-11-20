package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
)

// The one automatic rerun (reliability, layer 13). A hosted runner that is shut down while a
// job runs ends it with "The runner has received a shutdown signal" and exit code 143. That is
// the infrastructure's failure, not the change's, and a rerun of the same commit is the only
// remedy. rerun-on-runner-shutdown.yml calls `prcheck rerun` when a CI workflow run has failed.
//
// The rule is narrow on purpose:
//   - only a run that concluded "failure", and only its first attempt: a second failure of the
//     same commit is never rerun, so a real defect cannot be retried until it passes;
//   - only when every failed job carries a failure annotation with the shutdown text: one job
//     that failed for any other reason (a test, a lint, a gate) means no rerun at all;
//   - it says so every time, in the job summary and, for a pull request, in a comment: a rerun
//     is never silent, and neither is a refusal to rerun.
// No other workflow reruns anything automatically. A flaky test is fixed or reported, not retried.

const shutdownSignal = "The runner has received a shutdown signal"

// FailedJob is a job of the run that concluded "failure", with the messages of its
// failure-level annotations.
type FailedJob struct {
	Name        string
	Annotations []string
}

// DecideRerun reports whether a run should be rerun, and why or why not in one sentence.
func DecideRerun(conclusion string, attempt int, failed []FailedJob) (bool, string) {
	switch {
	case conclusion != "failure":
		return false, fmt.Sprintf("the run concluded %q, not failure; nothing to rerun", conclusion)
	case attempt != 1:
		return false, fmt.Sprintf("this is attempt %d; the one automatic rerun is spent, so a failure now is read as a failure", attempt)
	case len(failed) == 0:
		return false, "the run failed but no job concluded failure; not rerun"
	}
	for _, j := range failed {
		killed := false
		for _, a := range j.Annotations {
			killed = killed || strings.Contains(a, shutdownSignal)
		}
		if !killed {
			return false, fmt.Sprintf("job %q failed with no runner-shutdown annotation; a real failure is never rerun", j.Name)
		}
	}
	names := make([]string, len(failed))
	for i, j := range failed {
		names[i] = j.Name
	}
	return true, fmt.Sprintf("every failed job (%s) was ended by a runner shutdown signal; rerunning the failed jobs once", strings.Join(names, ", "))
}

type apiRun struct {
	Conclusion   string `json:"conclusion"`
	RunAttempt   int    `json:"run_attempt"`
	HTMLURL      string `json:"html_url"`
	Name         string `json:"name"`
	PullRequests []struct {
		Number int `json:"number"`
	} `json:"pull_requests"`
}

type apiJobs struct {
	Jobs []struct {
		ID         int64  `json:"id"`
		Name       string `json:"name"`
		Conclusion string `json:"conclusion"`
	} `json:"jobs"`
}

type apiAnnotation struct {
	Level   string `json:"annotation_level"`
	Message string `json:"message"`
}

func ghDo(args ...string) error {
	out, err := exec.Command("gh", args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("gh %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return nil
}

// rerunRun reads run runID of repo, decides, acts, and reports. It returns an error only when
// the API cannot be read or written.
func rerunRun(w io.Writer, summary, repo string, runID int64) error {
	var r apiRun
	if err := ghJSON(&r, "api", fmt.Sprintf("repos/%s/actions/runs/%d", repo, runID)); err != nil {
		return err
	}
	var jobs apiJobs
	if err := ghJSON(&jobs, "api", fmt.Sprintf("repos/%s/actions/runs/%d/jobs?per_page=100&filter=latest", repo, runID)); err != nil {
		return err
	}
	var failed []FailedJob
	for _, j := range jobs.Jobs {
		if j.Conclusion != "failure" {
			continue
		}
		var anns []apiAnnotation
		if err := ghJSON(&anns, "api", fmt.Sprintf("repos/%s/check-runs/%d/annotations", repo, j.ID)); err != nil {
			return err
		}
		f := FailedJob{Name: j.Name}
		for _, a := range anns {
			if a.Level == "failure" {
				f.Annotations = append(f.Annotations, a.Message)
			}
		}
		failed = append(failed, f)
	}
	rerun, why := DecideRerun(r.Conclusion, r.RunAttempt, failed)
	verb := "not rerun"
	if rerun {
		verb = "rerun once"
	}
	msg := fmt.Sprintf("%s run %s (attempt %d): %s. %s", r.Name, r.HTMLURL, r.RunAttempt, verb, why)
	_, _ = fmt.Fprintln(w, msg)
	if summary != "" {
		if f, err := os.OpenFile(summary, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o644); err == nil {
			_, _ = fmt.Fprintf(f, "### Rerun on runner shutdown\n\n%s\n", msg)
			_ = f.Close()
		}
	}
	if !rerun {
		return nil
	}
	if err := ghDo("api", "-X", "POST", fmt.Sprintf("repos/%s/actions/runs/%d/rerun-failed-jobs", repo, runID)); err != nil {
		return err
	}
	for _, pr := range r.PullRequests {
		body := fmt.Sprintf("The hosted runner was shut down under %s (%q); the failed jobs were rerun once, automatically. This is the only automatic rerun: a second failure is not rerun, and a failure without the shutdown message is never rerun.", r.HTMLURL, r.Name)
		if err := ghDo("api", "-X", "POST", fmt.Sprintf("repos/%s/issues/%d/comments", repo, pr.Number), "-f", "body="+body); err != nil {
			return err
		}
	}
	return nil
}
