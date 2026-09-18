// Package lora implements the LoRA (Low-Rank Adaptation) control-plane subsystem
// for BLIS: adapter identity, a pre-declared adapter registry (id → rank), a
// per-instance resident-adapter set with LRU eviction, and the three Digital-Twin
// adapter cost terms (cold-load latency, per-step compute overhead, HBM footprint).
//
// # No-op default (INV-6)
//
// The subsystem is inert unless adapters are configured. With no adapters declared
// and no capacity set, every code path reduces to a no-op and output is
// byte-identical to a pre-feature build. This is the load-bearing invariant for the
// whole feature (SC-001, SC-006).
//
// # Layering (Principle I)
//
// This package is a subpackage of sim/ and registers its capabilities into sim/ via
// init() — there is NO reverse import (sim/ never imports sim/lora). Cross-cutting
// fields that must live on sim/ bridge types (Request.Adapter, SimConfig.LoRAConfig,
// RoutingSnapshot resident-set) are defined in sim/, not here.
//
// # Determinism (Principle II)
//
// The subsystem introduces NO randomness. The resident-set LRU order and all cost
// terms are deterministic functions of arrival order, so no new PartitionedRNG
// subsystem is added. Per-adapter metric maps are key-sorted before output (R2).
package lora
