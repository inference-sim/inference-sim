// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/lora/worker_manager.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// LRUCacheWorkerLoRAManager is the class.
// Also vllm/lora/model_manager.py (LoRAModelManager, LRUCacheLoRAModelManager, AdapterLRUCache) and
// vllm/utils/cache.py (LRUCache).

// Package lora is the worker's LoRA adapter cache: the two LRUs vLLM's model runner drives once per
// scheduled batch. The scheduler never reads max_cpu_loras; this cache does, and an adapter missing
// from it is loaded before its batch runs, which is a cost the engine reports (Event).
package lora

import "fmt"

// Kind is what the cache did to one adapter.
type Kind string

const (
	// Load reads an adapter that is not in the CPU cache (max_cpu_loras) from its path
	// (LRUCacheWorkerLoRAManager._load_adapter).
	Load Kind = "load"
	// Evict removes the oldest adapter from the CPU cache to make room for a load
	// (LRUCacheLoRAModelManager.remove_oldest_adapter).
	Evict Kind = "evict"
	// Activate moves a cached adapter into a GPU slot (max_loras) so the batch can use it
	// (LoRAModelManager.activate_adapter).
	Activate Kind = "activate"
	// Deactivate frees an adapter's GPU slot, for a newcomer or because it left the CPU cache
	// (_deactivate_adapter).
	Deactivate Kind = "deactivate"
)

// Event is one action of a step, in the order vLLM takes it.
type Event struct {
	Kind    Kind
	Adapter int // lora_int_id
}

// lru is vLLM's LRUCache over adapter ids: oldest first. A key's use moves it to the end.
type lru struct{ keys []int }

func (l *lru) index(k int) int {
	for i, x := range l.keys {
		if x == k {
			return i
		}
	}
	return -1
}
func (l *lru) has(k int) bool { return l.index(k) >= 0 }

// touch is LRUCache.touch and the use of a key (__getitem__, get).
func (l *lru) touch(k int) {
	if i := l.index(k); i >= 0 {
		l.keys = append(append(l.keys[:i:i], l.keys[i+1:]...), k)
	}
}
func (l *lru) put(k int) { l.keys = append(l.keys, k) }
func (l *lru) remove(k int) {
	l.keys = append(l.keys[:l.index(k):l.index(k)], l.keys[l.index(k)+1:]...)
}
func (l *lru) oldest() (int, bool) {
	if len(l.keys) == 0 {
		return 0, false
	}
	return l.keys[0], true
}

// Cache is LRUCacheWorkerLoRAManager's state: the CPU cache of loaded adapters
// (capacity max_cpu_loras) and the GPU cache of adapters in slots (max_loras slots).
type Cache struct {
	capacity, slots int
	registered      lru   // _registered_adapters
	active          lru   // _active_adapters
	slotOf          []int // lora_index_to_id; 0 is a free slot (lora_int_id is positive)
}

// New is a cache of maxLoRAs GPU slots and maxCPULoRAs CPU entries. vLLM requires the CPU cache
// to hold at least the slots (LoRAModelManager.__init__, assert self.capacity >= self.lora_slots).
func New(maxLoRAs, maxCPULoRAs int) (*Cache, error) {
	if maxLoRAs < 1 {
		return nil, fmt.Errorf("lora: max_loras %d", maxLoRAs)
	}
	if maxCPULoRAs < maxLoRAs {
		return nil, fmt.Errorf("lora: max_cpu_loras %d is below max_loras %d", maxCPULoRAs, maxLoRAs)
	}
	return &Cache{capacity: maxCPULoRAs, slots: maxLoRAs, slotOf: make([]int, maxLoRAs)}, nil
}

// SetActive is set_active_adapters for one batch: every adapter the batch uses is made resident
// and given a slot, in the order given (duplicates count once, at their first place, as vLLM's
// dict of requests does). More adapters than slots is the error vLLM raises.
func (c *Cache) SetActive(adapters []int) ([]Event, error) {
	var order []int
	seen := map[int]bool{}
	for _, a := range adapters {
		if a < 1 {
			return nil, fmt.Errorf("lora: adapter id %d", a)
		}
		if !seen[a] {
			seen[a] = true
			order = append(order, a)
		}
	}
	if len(order) > c.slots {
		return nil, fmt.Errorf("lora: number of requested LoRAs (%d) is greater than the number of GPU LoRA slots (%d)",
			len(order), c.slots)
	}
	var ev []Event
	for _, a := range order {
		ev = c.add(a, ev)
	}
	return ev, nil
}

// add is LRUCacheWorkerLoRAManager.add_adapter.
func (c *Cache) add(a int, ev []Event) []Event {
	if !c.registered.has(a) {
		ev = append(ev, Event{Load, a})
		// Loading succeeded; now evict the oldest if the cache would exceed its capacity.
		if len(c.registered.keys)+1 > c.capacity {
			ev = c.removeOldestRegistered(ev)
		}
		c.registered.put(a)
	} else {
		// Already loaded: touch it to update its place (get_adapter).
		c.registered.touch(a)
	}
	return c.activate(a, ev)
}

// removeOldestRegistered is remove_oldest_adapter: the evicted adapter loses its GPU slot too
// (AdapterLRUCache._on_remove calls deactivate_adapter).
func (c *Cache) removeOldestRegistered(ev []Event) []Event {
	old, ok := c.registered.oldest()
	if !ok {
		return ev
	}
	c.registered.remove(old)
	if c.active.has(old) {
		ev = c.freeSlot(old, ev)
		c.active.remove(old)
	}
	return append(ev, Event{Evict, old})
}

// freeSlot is _deactivate_adapter.
func (c *Cache) freeSlot(a int, ev []Event) []Event {
	for i, id := range c.slotOf {
		if id == a {
			c.slotOf[i] = 0
			return append(ev, Event{Deactivate, a})
		}
	}
	return ev
}

// activate is LRUCacheLoRAModelManager.activate_adapter.
func (c *Cache) activate(a int, ev []Event) []Event {
	if !c.active.has(a) && len(c.active.keys) >= c.slots {
		old, _ := c.active.oldest()
		c.active.remove(old)
		ev = c.freeSlot(old, ev)
	}
	if !c.active.has(a) {
		// LoRAModelManager.activate_adapter: the first free slot; reading the adapter from the
		// CPU cache uses it, and the GPU cache records it.
		for i, id := range c.slotOf {
			if id == 0 {
				c.registered.touch(a)
				c.active.put(a)
				c.slotOf[i] = a
				ev = append(ev, Event{Activate, a})
				break
			}
		}
	}
	c.active.touch(a)
	return ev
}

// Registered is the CPU cache's adapters, oldest first.
func (c *Cache) Registered() []int { return append([]int(nil), c.registered.keys...) }

// Active is the GPU cache's adapters, oldest first.
func (c *Cache) Active() []int { return append([]int(nil), c.active.keys...) }

// Slots is lora_index_to_id: the adapter in each GPU slot, 0 for a free one.
func (c *Cache) Slots() []int { return append([]int(nil), c.slotOf...) }
