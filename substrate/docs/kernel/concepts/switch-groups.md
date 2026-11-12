# Switch groups

<p class="lede">Nodes under one leaf switch reach each other at their NIC rate. Traffic that leaves the
switch goes up one uplink, which is usually slower than the NICs below it. A scenario can state which
nodes sit under which switch and how oversubscribed the uplink is; the kernel then makes KV pulls
between groups share that uplink.</p>

This page follows design section 2.2 and SUB-3 as extended by blis-design #6
(30751c1a756769d7fadc4c0e030d34ccc12162ef): the cluster states which nodes sit under each switch and how
oversubscribed each group's uplink is. A fabric class in the catalog cannot carry it, because a grouping and a
ratio are one cluster's wiring, not a reusable class.

## Inputs

```yaml
cluster:
  hardware: h200
  fabric: ib-400g
  nodes: 12
  gpus_per_node: 8
  switch_groups: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]   # a partition of 0..nodes-1
  uplink_oversubscription: 4                            # at least 1
```

Pools take consecutive nodes in declaration order, so listing the groups as explicit node lists places a
pool in any combination of groups. `Cluster.Switching()` is the one function that reads the two fields.

## The rule

For a handoff between two nodes in different groups g (sender) and h (receiver), the
kernel keeps the two NIC links and adds `uplink/egress/group/g` and `uplink/ingress/group/h`. Each group's
uplink is offered to one rank's flow at

    C_g = n_g x gpusPerNode x nic / (r x rankGPUs)

bytes per second, where n_g is the group's node count, nic the effective rate of one rail, r the stated ratio
and rankGPUs the GPUs of one instance of the pool. The payload is one rank's, and the instance's rankGPUs ranks
pull at once through the same uplink, so one rank's flow gets 1/rankGPUs of it. With TP8 on 8-GPU nodes,
C_g = n_g x nic / r.

Consequences, each pinned by a test in `kernel/switch_groups_test.go`:

- a pull alone costs what it costs without groups, unless r is large enough that C_g falls below the NIC rate;
- two pulls that leave one group for another share both uplinks; when C_g equals the NIC rate they take twice as long as one;
- pulls inside a group name no uplink and do not slow each other;
- pulls between different pairs of groups share no uplink.

## Worked example

`ib-400g` at the released efficiency of 0.811 gives nic = 40.6 GB/s. Five nodes per group, r = 3.4, TP8: C_g = 59.7 GB/s.
One pull runs at the NIC rate, 40.6. Two pulls through the same uplink get 29.8 each; three, 19.9.

## What is not modelled

- Only KV pulls between pools are charged. A step collective that crosses a group boundary is priced at the node
  rate; the kernel states this as the assumption `switch_group_collectives` when an instance spans nodes.
- One link per node and direction, not one per rail.
- Up and down capacity are equal, routing inside a group is non-blocking, there is no background traffic, and
  one ratio applies to every group.
- A decode pool whose instances differ in size from the prefill pool's is priced as if its flows matched the prefill's.

## Reproduce

```sh
go test ./kernel/ -run 'CrossGroup|InGroup|LonePull|UplinksAre|Oversubscribed' -v
go test ./contracts/spec/scenario/ -run SwitchGroups -v
```

Both take seconds. The kernel tests share a link equally among the demands that name it, which is exact for the
equal pulls they pose; how a simulator divides a link among unequal transfers is the simulator's work.
The tests show the pricing rule, not a measured fabric: the ratio is whatever the scenario states.
