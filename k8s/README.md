# GitHub Actions Self-Hosted Runner on Kubernetes

Deploys the repo's self-hosted GitHub Actions runners as a Kubernetes Deployment
in the `blis` namespace. Runners are **ephemeral**: each container serves exactly
one job, deregisters, and is replaced with a fresh container.

`deployment.yaml` is reconciled against what is actually running. If you change
the live deployment with `kubectl patch`/`edit`, update the manifest in the same
change — a silently divergent manifest previously caused a disk-pressure incident
to be diagnosed against a topology that did not exist.

## Configuration

| Env var | Value | Description |
|---------|-------|-------------|
| `REPO_URL` | `https://github.com/inference-sim/inference-sim` | Repo the runner registers against |
| `RUNNER_NAME` | `fieldRef: metadata.name` | **Must stay per-pod.** See "Scaling" |
| `LABELS` | `self-hosted` | Matches `runs-on: [self-hosted]` in the workflows |
| `RUNNER_SCOPE` | `repo` | Repo-level (not org-level) registration |
| `EPHEMERAL` | `true` | One job per container. See "Ephemeral mode" |
| `DISABLE_AUTO_UPDATE` | `true` | Pins the runner to the image's version |
| `RUNNER_WORKDIR` | `/tmp/runner/_work` | On the `tmp` emptyDir |
| `ACCESS_TOKEN` | secret `github-runner-secret` key `github_token` | PAT |

The PAT needs **Administration (Read & Write)** on the target repo.

## Setup

### 1. Create the secret

The key **must** be `github_token` — that is what `deployment.yaml` reads.

```bash
kubectl create secret generic github-runner-secret \
  --namespace blis \
  --from-literal=github_token="$GITHUBACTIONS_RUNNER_TOKEN"
```

### 2. Deploy

```bash
kubectl apply -f k8s/deployment.yaml
```

### 3. Verify

```bash
kubectl -n blis get pods -l app=github-runner
kubectl -n blis logs -l app=github-runner --tail=20
gh api repos/inference-sim/inference-sim/actions/runners \
  --jq '.runners[] | "\(.name) \(.status) busy=\(.busy)"'
```

Each pod's log should end at `Listening for Jobs`, and each should appear
`online` in GitHub named after its pod.

## Ephemeral mode

The image entrypoint tests `[ -n "${EPHEMERAL}" ]` — **non-empty**, not
`== "true"`. Any value enables `--ephemeral`, *including the string `"false"`*.
To genuinely disable ephemeral mode you must **remove the variable**, not set it
to `"false"`.

Consequences, all relied on elsewhere in this doc:

- One job per container. `RESTARTS` climbing is **normal operation**, not a crash
  loop — it is roughly a job counter.
- Each restart gets a **fresh writable layer**, so Go build/module caches under
  `$HOME` (`/root`) never accumulate across jobs; measured at ~1 MB. No post-job
  cleanup hook is needed.
- Only `/tmp` (the emptyDir) survives a container restart, so `_work` checkouts
  persist for the pod's lifetime (~650 MB, bounded by `actions/checkout` reusing
  the clone).

`DISABLE_AUTO_UPDATE=true` matters *because* of ephemeral mode: without it the
runner self-updates on registration, exits to apply the update, and — since
ephemeral wipes `.runner` on exit — re-downloads the same ~200 MB update on the
next container, forever, charging it to the disk budget below each time. This was
observed as a pod restarting every ~4.5 minutes.

> **Trade-off:** pinning means the runner version now comes only from the image
> tag. GitHub eventually refuses connections from runners that are too old, so
> refresh the image periodically:
>
> ```bash
> kubectl -n blis set image deployment/github-runner \
>   runner=myoung34/github-runner:ubuntu-jammy
> kubectl -n blis rollout restart deployment/github-runner
> ```
>
> Check the running version against the latest release with:
>
> ```bash
> kubectl -n blis logs -l app=github-runner --tail=50 | grep 'runner version'
> gh api repos/actions/runner/releases/latest --jq .tag_name
> ```

## Disk Usage

Pods were being evicted with:

```
Pod ephemeral local storage usage exceeds the total limit of containers 4Gi.
```

This is **not** node DiskPressure, and **not** cache accumulation. The
`blislimits` LimitRange injects a *default* ephemeral-storage limit of **4Gi**
into any container that declares none, and baseline occupancy is already ~3 GB:

| Component | Size | Persists across restart? |
|-----------|------|--------------------------|
| Writable layer (`/actions-runner`, ~600 MB of it `externals`) | ~2.3 GB | No — fresh per job |
| `/tmp/runner/_work` (emptyDir) | ~650 MB | Yes — pod lifetime |

One `go test ./...` adds a ~2.5 GB build + module cache on top of that ~3 GB
baseline, exceeding 4Gi. So it is a **single-job peak**, which is why the fix is
a declared limit rather than a cleanup hook:

| Mechanism | Effect |
|-----------|--------|
| `resources.limits.ephemeral-storage: 12Gi` | Replaces the LimitRange's 4Gi default with room for the real peak |
| `resources.requests.ephemeral-storage: 2Gi` | Scheduler accounts for it; pod is eviction-safe from node DiskPressure while under the request |
| `volumes.tmp.emptyDir.sizeLimit: 6Gi` | Bounds the one path that survives restarts, and names the volume on overflow |

The LimitRange sets no `max`, and the ResourceQuota has no ephemeral-storage
entry, so raising the limit is unconstrained. Inspect them with:

```bash
kubectl -n blis describe limitrange blislimits
kubectl -n blis describe resourcequota blisquota
```

Check real usage in a live pod, and confirm evictions have stopped:

```bash
kubectl -n blis exec deploy/github-runner -- du -sm /actions-runner /tmp/runner /root
kubectl -n blis get events --field-selector reason=Evicted --sort-by=.lastTimestamp
```

The largest remaining lever is not in this manifest: the workflows that run
`go test ./...` and build `golangci-lint` on the self-hosted runner are what
create the peak. `ci.yml` already runs both on GitHub-hosted runners.

## Scaling

```bash
kubectl -n blis scale deployment github-runner --replicas=3
```

One pod runs one job at a time, so `replicas` is the number of concurrent lanes
for all agent work — which is why deliveries queue when it is low. Remember to
mirror the new value into `deployment.yaml`.

> **`RUNNER_NAME` must stay per-pod** (`fieldRef: metadata.name`). With a shared
> literal name, `config.sh --replace` makes each new pod replace the previous
> registration and invalidate the older listener's credentials, so replicas would
> evict each other. The cost of per-pod names is a stale `offline` registration
> in GitHub per replaced pod; ephemeral mode's deregister-on-exit clears most of
> them.

`maxUnavailable: 0` means a rolling update surges a new pod before terminating an
old one. A terminating pod still loses any job it was running, so prefer rolling
out while runners are idle:

```bash
gh api repos/inference-sim/inference-sim/actions/runners \
  --jq '[.runners[] | select(.busy)] | length'   # 0 = safe
```

## Updating the PAT

```bash
kubectl -n blis delete secret github-runner-secret
kubectl -n blis create secret generic github-runner-secret \
  --from-literal=github_token="$GITHUBACTIONS_RUNNER_TOKEN"
kubectl -n blis rollout restart deployment/github-runner
```

## Teardown

```bash
kubectl delete -f k8s/deployment.yaml
```

## How It Works

The `myoung34/github-runner:ubuntu-jammy` entrypoint fetches a registration
token from the GitHub API using `ACCESS_TOKEN`, registers the runner under its
pod name with `--ephemeral` and `--disableupdate`, and runs `Runner.Listener`.
After one job the runner deregisters and the process exits; the kubelet starts a
fresh container, which registers again. There is no init container and no
persistent runner home — `/tmp` is the only volume.
