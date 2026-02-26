# Design: `--roofline` Auto-Fetch Flag

**Status:** Approved
**Issue:** #414
**Type:** Specification
**Date:** 2026-02-25

## Problem

Running BLIS in roofline mode requires users to manually download HuggingFace `config.json` files and pass multiple explicit flags (`--model-config-folder`, `--hardware-config`). This is unnecessary friction since BLIS already knows the model name from `--model` and ships a bundled `hardware_config.json`.

## Decision

Add a `--roofline` boolean CLI flag. When set, BLIS automatically resolves both the HuggingFace model config and the hardware config without requiring `--model-config-folder` or `--hardware-config`.

**Desired UX:**
```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --roofline --hardware H100 --tp 1
```

## Design

### CLI Interface

**New flag:** `--roofline` (bool, default `false`, store-true)

**Required companions when `--roofline` is set:**
- `--model` (already enforced globally)
- `--hardware` (GPU type)
- `--tp` > 0

**Precedence rules:**
1. `--model-config-folder` (explicit) > `model_configs/` (local) > HF fetch (into `model_configs/`)
2. `--hardware-config` (explicit) > bundled `hardware_config.json`

Explicit overrides always win. The `--roofline` flag only activates the auto-resolution chain for whichever config is not explicitly provided.

### HuggingFace Config Resolution Chain

When `--roofline` is set and `--model-config-folder` is NOT provided, resolve `config.json` via (first success wins):

1. **Local `model_configs/`:** `model_configs/<model-short-name>/config.json` (previously fetched)
2. **HF HTTP fetch:** `GET https://huggingface.co/<hf_repo>/resolve/main/config.json`
   - Supports `HF_TOKEN` env var for gated models (`Authorization: Bearer <token>`)
   - On success, writes into `model_configs/<model-short-name>/` for future use
3. **Error:** Clear message listing what was tried

### Model Name to HF Repo Mapping

BLIS model names (lowercase, e.g., `meta-llama/llama-3.1-8b-instruct`) differ from HuggingFace repo names (mixed case, e.g., `meta-llama/Llama-3.1-8B-Instruct`). A `hf_repo` field in `defaults.yaml` maps BLIS names to HF repos:

```yaml
defaults:
  meta-llama/llama-3.1-8b-instruct:
    GPU: H100
    tensor_parallelism: 2
    vllm_version: vllm/vllm-openai:v0.8.4
    hf_repo: meta-llama/Llama-3.1-8B-Instruct
```

For models without `hf_repo`, fall back to using `--model` value directly as the HF repo path.

### Hardware Config Resolution

When `--roofline` is set and `--hardware-config` is NOT provided:
- Use bundled `hardware_config.json` at the default path (derived from `defaults.yaml` location)
- Error if the file is missing or the `--hardware` GPU is not found in it

### Integration with Existing Roofline Path

The existing implicit roofline detection (coefficients all-zero + explicit flags) remains unchanged. `--roofline` adds a new explicit activation path:

```
if --roofline is set:
  1. Validate --hardware and --tp are provided
  2. Resolve modelConfigFolder via chain (if not explicitly provided)
  3. Resolve hwConfigPath to bundled default (if not explicitly provided)
  4. Set roofline = true
  5. Continue to existing GetModelConfig() / GetHWConfig() calls
else:
  existing behavior unchanged
```

### File Structure

- **New:** `cmd/hfconfig.go` — HF config resolution chain, HTTP fetch, caching
- **New:** `cmd/hfconfig_test.go` — tests for resolution, caching, fallback, error cases
- **Modified:** `cmd/root.go` — `--roofline` flag registration and integration
- **Modified:** `cmd/default_config.go` — `GetHFRepo()` to read `hf_repo` from defaults
- **Modified:** `defaults.yaml` — add `hf_repo` field to model entries

### Error Messages

| Scenario | Message |
|----------|---------|
| HF 404 | `--roofline: model %q not found on HuggingFace (HTTP 404). Check --model or provide --model-config-folder` |
| HF 401 | `--roofline: model %q requires auth (HTTP 401). Set HF_TOKEN env var or provide --model-config-folder` |
| Network fail, no fallback | `--roofline: could not fetch config.json for %q, no cached or bundled config found` |
| Bundled hw config missing | `--roofline: bundled hardware config not found at %q. Provide --hardware-config` |
| Missing --hardware | `--roofline requires --hardware (GPU type)` |
| Missing --tp | `--roofline requires --tp > 0` |

## Boundaries

- All network I/O stays in `cmd/` (R6: no Fatalf in library, separation of concerns)
- `sim/` library is not modified
- Existing implicit roofline path is unchanged (backward compatible)
- No new external dependencies (stdlib `net/http` only)
