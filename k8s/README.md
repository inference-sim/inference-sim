# GitHub Actions Self-Hosted Runner on Kubernetes

Deploys a persistent GitHub Actions self-hosted runner as a Kubernetes pod for the Claude Code workflow. No cluster-admin access required.

## What This Does

The deployment runs a self-hosted GitHub Actions runner inside Kubernetes that picks up `@claude` mentions on issues and PRs. When a user comments `@claude` on an issue or PR, GitHub Actions triggers the Claude workflow (`.github/workflows/claude.yml`), which is routed to this runner via the `self-hosted` label.

The runner registers once with a fixed name (`k8s-blis`) and **stays alive** to handle jobs continuously. If the pod restarts (crash, node eviction, etc.), it detects the existing registration and re-registers with `--replace` without creating duplicates.

### Behavior with `@claude` mentions

- **Single mention**: The runner picks up the job, executes Claude, then returns to listening for the next job.
- **Multiple mentions in quick succession**: Jobs run sequentially on a single replica. Additional jobs queue in GitHub Actions until the current job finishes.
- **Concurrent users**: Each `@claude` mention triggers a separate workflow run. With 1 replica, jobs run one at a time. Scale replicas for concurrency (see [Scaling](#scaling)).

### OpenShift compatibility

The deployment includes workarounds for OpenShift's security constraints (forced non-root UID, read-only image layers). An init container copies runner binaries into a writable volume and patches the entrypoint to bypass UID checks.

## Prerequisites

- `kubectl` configured with access to your cluster
- A GitHub fine-grained PAT with Administration (Read & Write) scope
- Permission to create Deployments and Secrets in your namespace
- The cluster must have network access to your LiteLLM proxy endpoint

## Creating a GitHub PAT

The runner needs a Personal Access Token to register itself with GitHub. This PAT is **only** used for runner registration -- it does not access PRs, issues, or secrets (those use the built-in `GITHUB_TOKEN` provided by GitHub Actions).

### Option A: Fine-grained PAT (recommended)

1. Go to https://github.com/settings/tokens?type=beta (or Organization Settings > Developer settings > Fine-grained tokens)
2. Click **Generate new token**
3. Set:
   - **Token name**: `k8s-runner-registration`
   - **Repository access**: Select **Only select repositories** and choose `inference-sim/inference-sim`
   - **Permissions**:
     - **Administration**: Read and Write (required for runner registration)
   - No other permissions are needed
4. Click **Generate token** and save the value

### Option B: Classic PAT

1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Set:
   - **Note**: `k8s-runner-registration`
   - **Scopes**: `repo` (minimum required scope for classic PATs)
4. Click **Generate token** and save the value

Store the token in an environment variable (e.g., in `.bash_profile`):

```bash
export GITHUBACTIONS_RUNNER_TOKEN="ghp_your_token_here"
```

## Setup

### 1. Create the namespace

Follow instructions to create a namespace in the internal `etevpc-int-shared-us-east` cluster.

### 2. Create the secret

```bash
kubectl create secret generic github-runner-secret \
  --namespace blis \
  --from-literal=github_pat=$GITHUBACTIONS_RUNNER_TOKEN
```

### 3. Deploy the runner

```bash
kubectl apply -f k8s/deployment.yaml
```

### 4. Verify

```bash
kubectl -n blis get pods
kubectl -n blis logs -f deploy/github-runner
```

You should see output ending with:

```
Current runner version: '2.332.0'
Listening for Jobs
```

Confirm the runner appears at:
https://github.com/inference-sim/inference-sim/settings/actions/runners

## Scaling

With 1 replica (default), `@claude` jobs run one at a time. Increase replicas to handle concurrent jobs:

```bash
kubectl -n blis scale deployment github-runner --replicas=3
```

Each replica is an independent persistent runner. With 3 replicas, up to 3 `@claude` jobs can run simultaneously.

## Updating the PAT

If the PAT expires or is rotated:

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
