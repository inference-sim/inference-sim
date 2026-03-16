# GitHub Actions Self-Hosted Runner on Kubernetes

Deploys a persistent GitHub Actions self-hosted runner as a Kubernetes pod. Includes OpenShift compatibility (non-root UID, read-only image layers).

## Configuration

Edit `deployment.yaml` env vars to point at your repo:

| Env var | Default | Description |
|---------|---------|-------------|
| `GITHUB_REPO` | `inference-sim/inference-sim` | `owner/repo` for registration and repo URL |
| `RUNNER_NAME` | `k8s-runner` | Name shown in GitHub runner settings |
| `LABELS` | `self-hosted` | Runner labels for workflow targeting |
| `namespace` | `blis` | Kubernetes namespace (also update in kubectl commands below) |
| Deployment `name` | `github-runner` | Pod/deployment name — replace all occurrences of `github-runner` in the YAML |

The GitHub PAT must have **Administration (Read & Write)** scope on the target repo.

## Setup

### 1. Create a GitHub PAT

Fine-grained PAT (recommended): Settings > Developer settings > Fine-grained tokens > select your repo > Administration: Read and Write.

```bash
export GITHUBACTIONS_RUNNER_TOKEN="ghp_your_token_here"
```

### 2. Create the secret

```bash
kubectl create secret generic github-runner-secret \
  --namespace blis \
  --from-literal=github_pat=$GITHUBACTIONS_RUNNER_TOKEN
```

### 3. Deploy

```bash
kubectl apply -f k8s/deployment.yaml
```

### 4. Verify

```bash
kubectl -n blis logs -f deploy/github-runner
```

You should see `Listening for Jobs`. Confirm at `https://github.com/<owner>/<repo>/settings/actions/runners`.

## Scaling

```bash
kubectl -n blis scale deployment github-runner --replicas=3
```

## Updating the PAT

```bash
kubectl -n blis delete secret github-runner-secret
kubectl -n blis create secret generic github-runner-secret \
  --from-literal=github_pat=$GITHUBACTIONS_RUNNER_TOKEN
kubectl -n blis delete pod -l app=github-runner --force
```

## Teardown

```bash
kubectl delete -f k8s/deployment.yaml
kubectl delete -f k8s/secret.yaml
```

## How It Works

An **init container** copies runner binaries from the `myoung34/github-runner` image into a writable `emptyDir` volume and patches the entrypoint for OpenShift compatibility (non-root UID). The **main container** fetches a registration token via the GitHub API, registers once with `--replace`, then runs `Runner.Listener` to pick up jobs continuously. On pod restart, it detects the existing `.runner` config and skips re-registration.
