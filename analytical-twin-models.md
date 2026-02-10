# Analytical Twin Models for vLLM-Style LLM Inference Serving

## Model 1: Single-Stage M/G/1 Queue with Aggregate Service

**Intuition:**
Treat the entire LLM inference system as a single black-box server with an effective service time distribution. Each request is processed atomically from arrival to completion. This is the simplest possible queueing model and serves as a baseline for comparison.

**State Variables:**
- $N(t)$: Number of requests in system at time $t$
- $W(t)$: Workload (total remaining service time) at time $t$

**Formulation:**

*Arrival process:*
$$\text{Requests arrive as Poisson}(\lambda)$$

*Service time:*
$$S = \tau_{\text{eff}}(L_{\text{in}}, L_{\text{out}})$$
where $L_{\text{in}} \sim F_{\text{in}}$, $L_{\text{out}} \sim F_{\text{out}}$ are input/output token lengths.

*Effective service rate:*
$$\mu = \frac{1}{\mathbb{E}[S]} = \frac{1}{\mathbb{E}[\tau_{\text{eff}}]}$$

*Stability condition:*
$$\rho = \frac{\lambda}{\mu} < 1$$

*Conserved quantity (Work Conservation):*
$$\frac{dW}{dt} = \lambda \cdot \mathbb{E}[S] - \mathbf{1}_{W > 0}$$

In steady state: $\mathbb{E}[W] = \lambda \cdot \mathbb{E}[S^2] / (2(1-\rho))$ (Pollaczek-Khinchine).

*Mean metrics:*
$$\text{TTFT} + \text{TPOT} \cdot \mathbb{E}[L_{\text{out}}] = \mathbb{E}[W_q] + \mathbb{E}[S]$$

$$\text{Throughput} = \lambda \cdot \mathbb{E}[L_{\text{out}}] \quad \text{(tokens/sec, when stable)}$$

*Enforcement mechanism:* FIFO queueing disciplines enforce work conservation; the server drains work at unit rate when non-empty.

**Improvement over Model 0 (no model):**
Provides first quantitative predictions for latency and throughput under load.

**Captures:**
- Mean latency scaling with load
- Stability boundary ($\rho < 1$)
- Little's Law relationships

**Limitations:**
- No distinction between prefill and decode phases
- No batching effects
- No KV-cache constraints
- Treats service time as load-independent
- Cannot predict TTFT separately from total latency

---

## Model 2: Two-Stage Tandem Queue (Prefill → Decode)

**Intuition:**
Decompose inference into two distinct phases: prefill (compute-bound, processes input) and decode (memory-bandwidth-bound, generates output tokens). Model as a tandem queue where requests flow through both stages sequentially.

**State Variables:**
- $N_p(t)$: Number of requests in prefill stage
- $N_d(t)$: Number of requests in decode stage
- $W_p(t), W_d(t)$: Workloads at each stage

**Formulation:**

*Stage 1 (Prefill):*
$$S_p = \tau_p(L_{\text{in}}) = \alpha_p \cdot L_{\text{in}}$$
$$\mu_p = \frac{1}{\mathbb{E}[S_p]} = \frac{1}{\alpha_p \cdot \mathbb{E}[L_{\text{in}}]}$$

*Stage 2 (Decode):*
$$S_d = \tau_d \cdot L_{\text{out}}$$
$$\mu_d = \frac{1}{\mathbb{E}[S_d]} = \frac{1}{\tau_d \cdot \mathbb{E}[L_{\text{out}}]}$$

*Tandem queue dynamics:*
$$\frac{dN_p}{dt} = \lambda - \mu_p \cdot \mathbf{1}_{N_p > 0}$$
$$\frac{dN_d}{dt} = \mu_p \cdot \mathbf{1}_{N_p > 0} - \mu_d \cdot \mathbf{1}_{N_d > 0}$$

*Conserved quantity (Flow Conservation):*
In steady state, throughput through both stages equals:
$$\theta = \lambda = \mu_p \cdot \mathbb{P}[N_p > 0] = \mu_d \cdot \mathbb{P}[N_d > 0]$$

*Bottleneck identification:*
$$\mu_{\text{bottleneck}} = \min(\mu_p, \mu_d)$$

*Stability:* $\lambda < \mu_{\text{bottleneck}}$

*Metrics (assuming independence via Burke's theorem approximation):*
$$\text{TTFT} = \mathbb{E}[W_{q,p}] + \mathbb{E}[S_p]$$
$$\text{TPOT} = \frac{\mathbb{E}[W_{q,d}] + \mathbb{E}[S_d]}{\mathbb{E}[L_{\text{out}}]}$$

*Enforcement mechanism:* Each stage operates as independent M/G/1; flow conservation enforced by tandem coupling.

**Improvement over Model 1:**
Separates prefill from decode, enabling distinct TTFT and TPOT predictions.

**Captures:**
- Prefill/decode phase distinction
- TTFT as primarily prefill-driven
- TPOT as primarily decode-driven
- Bottleneck identification (compute vs. memory-bound)

**Limitations:**
- Assumes stages are independent (ignores continuous batching)
- No batch-dependent service rates
- Prefill and decode don't share resources realistically
- No KV-cache modeling

---

## Model 3: Fluid Queue with Fixed Batch Size

**Intuition:**
Replace discrete request dynamics with continuous fluid flow. Introduce batching by assuming a fixed batch size $B$ that processes multiple requests simultaneously, amortizing overhead and improving throughput.

**State Variables:**
- $q_p(t)$: Fluid level (requests) in prefill queue
- $q_d(t)$: Fluid level in decode queue
- $B$: Fixed batch size (parameter)

**Formulation:**

*Fluid dynamics:*
$$\frac{dq_p}{dt} = \lambda - \mu_p(B) \cdot \mathbf{1}_{q_p > 0}$$
$$\frac{dq_d}{dt} = \mu_p(B) \cdot \mathbf{1}_{q_p > 0} - \mu_d(B) \cdot \mathbf{1}_{q_d > 0}$$

*Batched service rates:*
$$\mu_p(B) = \frac{B}{\tau_p^{\text{batch}}(B)} = \frac{B}{\alpha_p \cdot \mathbb{E}[L_{\text{in}}] + \beta_p \cdot B}$$

$$\mu_d(B) = \frac{B}{\tau_d^{\text{batch}}(B)} = \frac{B}{\tau_d + \beta_d \cdot B}$$

Here $\beta_p, \beta_d$ capture per-request overhead within a batch.

*Conserved quantity (Fluid Work Conservation):*
$$\frac{d}{dt}\left(q_p + q_d\right) = \lambda - \mu_d(B) \cdot \mathbf{1}_{q_d > 0 \text{ or } q_p > 0}$$

Total fluid drains at the decode rate when system non-empty.

*Steady-state fluid levels:*
When $\lambda < \mu_d(B)$:
$$\bar{q}_p = \frac{\lambda}{\mu_p(B) - \lambda} \cdot \frac{\lambda}{\mu_p(B)}$$
$$\bar{q}_d = \frac{\lambda}{\mu_d(B) - \lambda} \cdot \frac{\lambda}{\mu_d(B)}$$

*Metrics:*
$$\text{TTFT} = \frac{\bar{q}_p}{\lambda} + \frac{\mathbb{E}[L_{\text{in}}]}{\mu_p(B)/B}$$
$$\text{Throughput} = \min\left(\lambda, \mu_d(B)\right) \cdot \mathbb{E}[L_{\text{out}}]$$

*Enforcement mechanism:* Fluid conservation; batches formed when $q_p \geq B$ or timeout.

**Improvement over Model 2:**
Introduces batching with throughput gains from parallel processing.

**Captures:**
- Batching efficiency (sublinear scaling of batch time with $B$)
- Throughput improvement from larger batches
- Tradeoff between latency (waiting to fill batch) and throughput

**Limitations:**
- Fixed batch size unrealistic (vLLM uses dynamic batching)
- No state-dependent service rates
- Fluid ignores discreteness and variance
- No KV-cache constraints

---

## Model 4: State-Dependent Batch Service Rate (Decode Focus)

**Intuition:**
The decode phase service rate depends critically on the current batch size, which varies dynamically. Model decode service as a state-dependent rate $\mu_d(B(t))$ where $B(t)$ is the instantaneous batch size (number of concurrent decoding requests).

**State Variables:**
- $N_p(t)$: Requests in prefill queue
- $B(t)$: Current decode batch size (requests actively decoding)
- $n_d(t)$: Requests waiting to join decode batch

**Formulation:**

*State-dependent decode service rate:*
$$\mu_d(B) = \frac{B}{\tau_{\text{step}}(B)}$$

where step time follows (empirically fitted):
$$\tau_{\text{step}}(B) = \tau_0 + \gamma \cdot B + \delta \cdot B^2$$

The quadratic term captures memory bandwidth saturation.

*Per-request token generation rate within batch:*
$$r(B) = \frac{1}{\tau_{\text{step}}(B)}$$

*Batch completion dynamics:*
Requests leave decode at rate:
$$\mu_{\text{complete}}(B) = B \cdot r(B) \cdot p_{\text{done}}$$
where $p_{\text{done}} = 1/\mathbb{E}[L_{\text{out}}]$ is geometric completion probability per step.

*Conserved quantity (Step Budget Conservation):*
Each decode step consumes one step budget unit. The system processes:
$$\text{Steps/sec} = \frac{1}{\tau_{\text{step}}(B)}$$
Total tokens generated per second (throughput):
$$\Theta = B \cdot r(B) = \frac{B}{\tau_{\text{step}}(B)}$$

*Batch size evolution (continuous approximation):*
$$\frac{dB}{dt} = \mu_p \cdot \mathbf{1}_{N_p > 0} - \mu_{\text{complete}}(B)$$

*Steady-state batch size $\bar{B}$ satisfies:*
$$\lambda = \bar{B} \cdot r(\bar{B}) \cdot p_{\text{done}} = \frac{\bar{B}}{\tau_{\text{step}}(\bar{B}) \cdot \mathbb{E}[L_{\text{out}}]}$$

*Metrics:*
$$\text{TPOT} = \tau_{\text{step}}(\bar{B})$$
$$\text{Throughput} = \frac{\bar{B}}{\tau_{\text{step}}(\bar{B})}$$

*Enforcement mechanism:* Batch size self-regulates via the fixed-point equation above.

**Improvement over Model 3:**
Service rate now depends on system state (batch size), capturing the fundamental throughput-latency tradeoff.

**Captures:**
- Throughput scaling with batch size
- TPOT degradation under load (larger batches → slower steps)
- Self-consistent equilibrium batch size
- Memory bandwidth saturation effects

**Limitations:**
- Prefill still treated simplistically
- No interaction between prefill and decode resource usage
- No KV-cache memory limits
- Ignores variance in batch size

---

## Model 5: Continuous Batching with Prefill-Decode Interleaving

**Intuition:**
vLLM uses continuous batching: new prefills can be inserted into ongoing decode batches, and the system alternates between prefill and decode operations within each "step." Model the joint dynamics of prefill insertions and decode iterations.

**State Variables:**
- $Q(t)$: Requests waiting in prefill queue
- $B(t)$: Active decode batch size
- $\phi(t)$: Fraction of step budget allocated to prefill

**Formulation:**

*Step decomposition:*
Each iteration step includes both prefill and decode work:
$$\tau_{\text{step}}(B, n_p) = \tau_d(B) + \tau_p(n_p)$$
where $n_p$ is number of new prefills inserted this step.

*Chunked prefill model:*
$$\tau_p(n_p) = \alpha_p \cdot \sum_{i=1}^{n_p} \min(L_{\text{in}}^{(i)}, C_{\text{chunk}})$$
where $C_{\text{chunk}}$ is the prefill chunk size.

*Resource sharing constraint (Step Time Budget):*
$$\tau_{\text{step}} \leq T_{\text{SLO}}^{\text{step}}$$

*Conserved quantity (Iteration Budget Conservation):*
Within each step of duration $\tau_{\text{step}}$:
- Decode tokens generated: $B$
- Prefill tokens processed: $n_p \cdot C_{\text{chunk}}$ (or full input if smaller)

Total compute consumed = Total compute available:
$$\text{FLOPs}_{\text{decode}}(B) + \text{FLOPs}_{\text{prefill}}(n_p) = \text{FLOPs}_{\text{available}}(\tau_{\text{step}})$$

*Admission control for prefill:*
$$n_p^* = \underset{n_p}{\arg\max} \; n_p \quad \text{s.t.} \quad \tau_{\text{step}}(B, n_p) \leq T_{\text{target}}$$

*Dynamics:*
$$\frac{dQ}{dt} = \lambda - \frac{n_p^*(B)}{\tau_{\text{step}}}$$
$$\frac{dB}{dt} = \frac{n_p^*(B)}{\tau_{\text{step}}} - \frac{B}{\tau_{\text{step}} \cdot \mathbb{E}[L_{\text{out}}]}$$

*Metrics:*
$$\text{TTFT} = \frac{Q}{\lambda} + \mathbb{E}\left[\frac{L_{\text{in}}}{C_{\text{chunk}}}\right] \cdot \tau_{\text{step}}$$
$$\text{TPOT} = \tau_{\text{step}}(B, n_p^*)$$

*Enforcement mechanism:* Scheduler enforces step time budget by limiting prefill insertion.

**Improvement over Model 4:**
Models continuous batching with prefill-decode interleaving within each step.

**Captures:**
- Chunked prefill behavior
- Prefill/decode resource competition
- Step time SLO enforcement
- Realistic vLLM scheduling dynamics

**Limitations:**
- No KV-cache memory constraints
- No preemption modeling
- Ignores prompt length variability effects on chunking
- Deterministic dynamics (no variance)

---

## Model 6: Batch-Size-Dependent Service Network

**Intuition:**
Model the system as a closed service network where the decode batch constitutes a finite population. Requests circulate: arrive → prefill → decode (multiple iterations) → complete. Service rates at each station depend on local population.

**State Variables:**
- $N_q$: Requests in arrival queue (waiting for prefill)
- $N_p$: Requests being prefilled (≤ max prefill concurrency)
- $N_d$: Requests in decode batch
- Total in system: $N = N_q + N_p + N_d$

**Formulation:**

*Service rates (state-dependent):*
$$\mu_q = \lambda \quad \text{(external arrivals)}$$
$$\mu_p(N_p, N_d) = \frac{N_p}{\tau_p(N_p) + \tau_d(N_d)} \cdot \mathbf{1}_{N_p > 0}$$
$$\mu_d(N_d) = \frac{N_d}{\tau_d(N_d) \cdot \mathbb{E}[L_{\text{out}}]} \quad \text{(completion rate)}$$

*Transition rates:*
- Queue → Prefill: $\min(\lambda, \mu_p^{\max})$
- Prefill → Decode: $\mu_p(N_p, N_d)$
- Decode → Exit: $\mu_d(N_d)$

*Conserved quantity (Population Conservation in Decode):*
Let $\pi(N_d)$ be steady-state distribution of decode batch size. Then:
$$\sum_{n=0}^{N_{\max}} n \cdot \pi(n) = \bar{N}_d$$
and flow balance requires:
$$\lambda = \bar{N}_d \cdot \frac{1}{\tau_d(\bar{N}_d) \cdot \mathbb{E}[L_{\text{out}}]}$$

*Quasi-Birth-Death Process for $N_d$:*
$$\pi(n+1) = \pi(n) \cdot \frac{\text{prefill completion rate}}{\text{decode completion rate}(n+1)}$$

*Effective throughput:*
$$\Theta = \sum_{n=1}^{N_{\max}} \pi(n) \cdot \frac{n}{\tau_d(n)}$$

*Metrics:*
$$\mathbb{E}[\text{TPOT}] = \sum_{n=1}^{N_{\max}} \pi(n) \cdot \tau_d(n)$$
$$\text{Var}[\text{TPOT}] = \sum_{n} \pi(n) \cdot \tau_d(n)^2 - \mathbb{E}[\text{TPOT}]^2$$

*Enforcement mechanism:* Birth-death balance enforces population conservation; service network structure enforces flow.

**Improvement over Model 5:**
Provides distributional predictions via the QBD process, not just means.

**Captures:**
- Batch size distribution (not just mean)
- TPOT variance from batch size fluctuations
- Service network structure
- Interaction between prefill and decode populations

**Limitations:**
- No KV-cache memory constraints
- No preemption
- Assumes geometric output lengths
- No explicit memory modeling

---

## Model 7: KV-Cache Capacity Constraint with Admission Control

**Intuition:**
The GPU has finite KV-cache memory $K$ (in tokens). Each active request consumes KV slots proportional to its current sequence length. Model memory as a hard constraint that limits admission of new requests.

**State Variables:**
- $N_d(t)$: Number of requests in decode
- $M(t)$: Total KV-cache memory used (tokens)
- $\ell_i(t)$: Current sequence length of request $i$ in decode
- $M(t) = \sum_{i \in \text{decode}} \ell_i(t)$

**Formulation:**

*KV-cache dynamics:*
$$\frac{dM}{dt} = \frac{N_d}{\tau_d(N_d)} - \sum_{i: \text{completing}} \ell_i$$

Each step, decode batch generates $N_d$ new tokens, growing memory.

*Memory constraint:*
$$M(t) \leq K$$

*Admission control:*
New request with input length $\ell_{\text{in}}$ admitted only if:
$$M(t) + \ell_{\text{in}} + \mathbb{E}[L_{\text{out}}] \leq K$$

(Conservative: reserve expected output space.)

*Effective admission rate:*
$$\lambda_{\text{eff}}(M) = \lambda \cdot \mathbf{1}_{M + \mathbb{E}[\ell_{\text{new}}] \leq K}$$

*Conserved quantity (Memory Conservation):*
$$M(t) = M(0) + \int_0^t \left[\frac{N_d(s)}{\tau_d(N_d(s))} - \text{completion outflow}(s)\right] ds$$

Memory is conserved: it grows with token generation and shrinks with request completion.

*Throughput under memory limit:*
$$\Theta_{\max} = \frac{K}{\mathbb{E}[L_{\text{in}}] + \mathbb{E}[L_{\text{out}}]} \cdot \frac{1}{\mathbb{E}[L_{\text{out}}] \cdot \tau_d}$$

*Knee behavior:*
Define memory utilization $u = M/K$.

For $u < u^*$ (knee): System is compute-bound, throughput scales with $\lambda$.
For $u \geq u^*$: System is memory-bound, throughput saturates.

*Metrics:*
$$\text{TTFT}(u) = \text{TTFT}_0 \cdot (1 + \kappa \cdot \mathbf{1}_{u > u^*})$$

where $\kappa$ captures queueing delay when memory-bound.

*Enforcement mechanism:* Admission control enforces $M \leq K$; memory conservation tracked via aggregate dynamics.

**Improvement over Model 6:**
First model with explicit memory constraints and admission control.

**Captures:**
- KV-cache capacity limits
- Memory-induced throughput saturation (knee)
- Admission control effects on TTFT
- Memory vs. compute bottleneck regimes

**Limitations:**
- No preemption (requests never evicted once admitted)
- No partial completion handling
- Aggregate memory model ignores per-request dynamics
- No cliff behavior from memory pressure

---

## Model 8: Preemption and Eviction Dynamics

**Intuition:**
When memory pressure is high, vLLM may preempt (pause and swap out) or evict (abort) requests to make room for new prefills. Model preemption as a state-dependent rate that activates under memory pressure.

**State Variables:**
- $N_a(t)$: Active requests (in GPU memory)
- $N_s(t)$: Swapped requests (in CPU memory)
- $M(t)$: GPU KV-cache usage
- $M_s(t)$: CPU swap space usage

**Formulation:**

*Preemption trigger:*
When a new prefill requires memory $m_{\text{new}}$ and $M + m_{\text{new}} > K$:
$$\text{Preempt requests until } M + m_{\text{new}} \leq K$$

*Preemption rate (state-dependent):*
$$\nu_{\text{preempt}}(M) = \nu_0 \cdot \left(\frac{M - K_{\text{thresh}}}{K - K_{\text{thresh}}}\right)^+ \cdot \lambda$$

where $(x)^+ = \max(0, x)$ and $K_{\text{thresh}}$ is preemption threshold.

*Swap-in rate:*
When $M < K_{\text{thresh}}$ and $N_s > 0$:
$$\nu_{\text{swap-in}} = \frac{N_s}{\tau_{\text{swap}}}$$

*Request lifetime with preemption:*
$$T_{\text{total}} = T_{\text{active}} + T_{\text{swapped}}$$
$$\mathbb{E}[T_{\text{total}}] = \mathbb{E}[T_{\text{active}}] \cdot (1 + \mathbb{E}[\text{preemption count}])$$

*Conserved quantity (Total Request Conservation):*
$$N_a(t) + N_s(t) + N_q(t) = N_{\text{total}}(t)$$
$$\frac{dN_{\text{total}}}{dt} = \lambda - \mu_{\text{complete}}$$

*Cliff behavior model:*
Define stress $\sigma = \lambda / \lambda_{\text{crit}}$ where $\lambda_{\text{crit}}$ is sustainable rate.

For $\sigma > 1$:
$$\text{Preemption rate} \propto (\sigma - 1)^{\alpha}$$
$$\text{Throughput collapse: } \Theta(\sigma) = \Theta_{\max} \cdot e^{-\beta(\sigma-1)^2}$$

*Metrics with preemption:*
$$\text{TTFT} = \text{TTFT}_0 + \mathbb{E}[\text{preemption wait}] \cdot p_{\text{preempt}}$$
$$\text{TPOT} = \text{TPOT}_0 \cdot (1 + \eta \cdot p_{\text{preempt}})$$

where $p_{\text{preempt}}(M) = \mathbb{P}[\text{request gets preempted} | M]$.

*Enforcement mechanism:* Preemption enforces memory constraint $M \leq K$; swap queue buffers preempted requests.

**Improvement over Model 7:**
Models preemption/eviction and the resulting latency impact and throughput cliffs.

**Captures:**
- Preemption dynamics under memory pressure
- Latency inflation from swap delays
- Throughput cliff at high load
- Hysteresis in recovery from overload

**Limitations:**
- Preemption rate parameters require fitting
- No detailed swap overhead modeling
- Aggregate treatment (no per-request tracking)
- No distributional predictions for latency under preemption

---

## Model 9: Mean-Field Fixed Point with Reflected Resource Constraints

**Intuition:**
Formulate the system as a mean-field model where aggregate state variables evolve according to ordinary differential equations, with resource constraints enforced via reflection (instantaneous correction when hitting boundaries). Find the steady-state operating point as a fixed point of the mean-field dynamics.

**State Variables:**
- $\bar{q}$: Mean queue length
- $\bar{b}$: Mean batch size
- $\bar{m}$: Mean memory utilization ($M/K$)
- $\bar{\ell}$: Mean sequence length in decode batch

**Formulation:**

*Mean-field ODEs:*

$$\frac{d\bar{q}}{dt} = \lambda - \mu_{\text{admit}}(\bar{m}) \cdot \mathbf{1}_{\bar{q} > 0}$$

$$\frac{d\bar{b}}{dt} = \mu_{\text{admit}}(\bar{m}) - \frac{\bar{b}}{\tau_d(\bar{b}) \cdot \mathbb{E}[L_{\text{out}}]}$$

$$\frac{d\bar{m}}{dt} = \frac{\bar{b}}{K \cdot \tau_d(\bar{b})} - \frac{\bar{b} \cdot \bar{\ell}}{K \cdot \tau_d(\bar{b}) \cdot \mathbb{E}[L_{\text{out}}]} + \nu_{\text{reflect}}(\bar{m})$$

*Reflected boundary for memory:*
$$\nu_{\text{reflect}}(\bar{m}) = -\left(\frac{d\bar{m}}{dt}\right)^+ \cdot \mathbf{1}_{\bar{m} = 1}$$

This is the Skorokhod reflection that prevents $\bar{m}$ from exceeding 1.

*Conserved quantity (Fluid Mass Conservation):*
$$\frac{d}{dt}(\bar{q} + \bar{b}) = \lambda - \frac{\bar{b}}{\tau_d(\bar{b}) \cdot \mathbb{E}[L_{\text{out}}]}$$

In steady state:
$$\lambda = \frac{\bar{b}^*}{\tau_d(\bar{b}^*) \cdot \mathbb{E}[L_{\text{out}}]}$$

*Fixed point equations:*
Setting all derivatives to zero:

$$\bar{b}^* = \lambda \cdot \tau_d(\bar{b}^*) \cdot \mathbb{E}[L_{\text{out}}] \quad \text{(implicit equation)}$$

$$\bar{m}^* = \frac{\bar{b}^* \cdot \mathbb{E}[\ell]}{K}$$

$$\bar{q}^* = \begin{cases} 0 & \text{if } \bar{m}^* < 1 \\ \frac{\lambda - \lambda_{\text{crit}}}{\mu_{\text{drain}}} & \text{if } \bar{m}^* = 1 \end{cases}$$

*Metrics at fixed point:*
$$\text{TTFT}^* = \frac{\bar{q}^*}{\lambda} + \frac{\mathbb{E}[L_{\text{in}}]}{\mu_p}$$
$$\text{TPOT}^* = \tau_d(\bar{b}^*)$$
$$\text{Throughput}^* = \frac{\bar{b}^*}{\tau_d(\bar{b}^*)}$$

*Linear stability analysis around fixed point:*
Jacobian $J = \partial \mathbf{f} / \partial (\bar{q}, \bar{b}, \bar{m})$ evaluated at $(\bar{q}^*, \bar{b}^*, \bar{m}^*)$.
System stable iff all eigenvalues of $J$ have negative real parts.

*Enforcement mechanism:* Reflection at $\bar{m}=1$ enforces memory conservation; fixed point represents dynamic equilibrium.

**Improvement over Model 8:**
Provides self-consistent equilibrium predictions via fixed-point analysis; reflection formalizes resource constraints.

**Captures:**
- Self-consistent operating point
- Stability/instability regimes
- Phase transitions (compute-bound ↔ memory-bound)
- Asymptotic behavior under scaling

**Limitations:**
- Mean-field: no variance or distributional predictions
- Reflection is approximate (true constraint is discrete)
- Fixed point may not be unique
- No transient dynamics

---

## Model 10: State-Dependent Service Network with KV Reflection and Diffusion Approximation

**Intuition:**
The full analytical twin: a state-dependent service network with explicit memory reflection, where diffusion approximations provide distributional predictions. Unifies all previous models as limiting cases. The system is characterized by a multi-dimensional state process with reflecting boundaries, and steady-state distributions are obtained via generator equations.

**State Variables (Full State Vector):**
$$\mathbf{X}(t) = (Q(t), B(t), M(t), \mathbf{L}(t))$$

where:
- $Q(t) \in \mathbb{R}_{\geq 0}$: Queue length (fluid)
- $B(t) \in \mathbb{Z}_{\geq 0}$: Decode batch size
- $M(t) \in [0, K]$: KV-cache memory usage
- $\mathbf{L}(t) = (L_1, \ldots, L_B)$: Sequence lengths (summarized by empirical distribution)

**Formulation:**

*State space:*
$$\mathcal{S} = \{(q, b, m, \boldsymbol{\ell}) : q \geq 0, b \geq 0, m \in [0, K], \|\boldsymbol{\ell}\|_1 = m\}$$

*Generator (infinitesimal dynamics):*

For test function $f$:
$$\mathcal{L}f(\mathbf{x}) = \underbrace{\lambda \cdot \partial_q f}_{\text{arrivals}} + \underbrace{\mu_p(b,m) \cdot [f(\mathbf{x}^{+p}) - f(\mathbf{x})]}_{\text{prefill completions}} + \underbrace{\sum_{i=1}^b r_i(b) \cdot [f(\mathbf{x}^{+d}_i) - f(\mathbf{x})]}_{\text{decode steps}}$$
$$+ \underbrace{\sum_{i=1}^b \mu_c^{(i)}(\ell_i) \cdot [f(\mathbf{x}^{-i}) - f(\mathbf{x})]}_{\text{completions}} + \underbrace{\nu_{\text{preempt}}(m) \cdot [f(\mathbf{x}^{\text{pre}}) - f(\mathbf{x})]}_{\text{preemption}}$$

*State-dependent rates:*

$$\mu_p(b, m) = \frac{\mu_p^0}{\tau_p + \gamma_p \cdot b} \cdot \mathbf{1}_{m + \mathbb{E}[L_{\text{in}}] \leq K}$$

$$r(b) = \frac{1}{\tau_d(b)} = \frac{1}{\tau_0 + \gamma_d b + \delta_d b^2}$$

$$\mu_c^{(i)}(\ell_i) = r(b) \cdot p_{\text{done}}(\ell_i) = r(b) \cdot \frac{1}{\mathbb{E}[L_{\text{out}}]}$$

$$\nu_{\text{preempt}}(m) = \nu_0 \cdot \left(1 - \frac{K - m}{K - K_{\text{thresh}}}\right)^+ \cdot \lambda$$

*Conserved quantities:*

**1. Work Conservation:**
$$\frac{d}{dt}\mathbb{E}[Q + B \cdot \mathbb{E}[L_{\text{out}}] \cdot \tau_d(B)] = \lambda \cdot \mathbb{E}[L_{\text{out}}] \cdot \tau_d^{\text{eff}} - \mathbf{1}_{\text{system active}}$$

**2. Memory Conservation (with reflection):**
$$M(t) = M(0) + \int_0^t g(B(s), M(s)) ds + R(t)$$
where $g$ is the drift and $R(t)$ is the reflection process:
$$R(t) = -\inf_{s \leq t} \left(M(0) + \int_0^s g(B(u), M(u)) du \right) \wedge 0 \vee (K - M(0) - \int_0^t g)$$

**3. Flow Conservation (Little's Law):**
$$\bar{N} = \lambda \cdot \bar{T}$$
$$\text{Throughput} \cdot \bar{T} = \bar{B} \cdot \mathbb{E}[L_{\text{out}}]$$

*Diffusion approximation (heavy traffic):*

Under scaling $\lambda^{(n)} = \lambda_c - \epsilon_n$, $\epsilon_n \to 0$:

$$\hat{\mathbf{X}}^{(n)}(t) = \sqrt{n}(\mathbf{X}^{(n)}(t) - \bar{\mathbf{x}}^*)$$

converges to reflected Ornstein-Uhlenbeck process:

$$d\hat{\mathbf{X}} = A(\hat{\mathbf{X}}) \cdot \hat{\mathbf{X}} \, dt + \Sigma^{1/2} \, dW(t) + d\mathbf{R}(t)$$

where:
- $A$ is drift matrix (Jacobian of mean-field at fixed point)
- $\Sigma$ is diffusion matrix (from variance of jumps)
- $\mathbf{R}(t)$ is vector reflection process at boundaries

*Stationary distribution:*

The stationary density $\pi(\mathbf{x})$ satisfies the Fokker-Planck equation:
$$\mathcal{L}^* \pi = 0$$
with reflecting boundary conditions:
$$\mathbf{n} \cdot (\mathbf{A}\pi - \nabla \cdot (\Sigma \pi)) = 0 \text{ on } \partial \mathcal{S}$$

*Closed-form approximation for marginals:*

$$\pi_B(b) \approx \text{Truncated-Poisson}(\bar{b}^*, \sigma_b^2) \text{ on } [0, B_{\max}]$$

$$\pi_M(m) \approx \text{Truncated-Normal}(\bar{m}^*, \sigma_m^2) \text{ on } [0, K]$$

with parameters from diffusion:
$$\sigma_b^2 = \frac{\text{Var}[\text{batch jumps}]}{2|a_{bb}|}$$
$$\sigma_m^2 = \frac{\text{Var}[\text{memory jumps}]}{2|a_{mm}|}$$

*Unified metric formulae:*

$$\text{TTFT} = \underbrace{\frac{\mathbb{E}[Q]}{\lambda}}_{\text{queueing}} + \underbrace{\mathbb{E}\left[\frac{L_{\text{in}}}{\mu_p(B,M)}\right]}_{\text{prefill}} + \underbrace{\mathbb{E}[\tau_{\text{preempt}} \cdot \mathbf{1}_{\text{preempted}}]}_{\text{preemption delay}}$$

$$\text{TPOT} = \mathbb{E}\left[\tau_d(B) \cdot \left(1 + \frac{\text{Var}[B]}{\bar{B}^2} \cdot \frac{\partial \tau_d}{\partial B}\Big|_{\bar{B}}\right)\right]$$

$$\text{Var}[\text{TPOT}] = \text{Var}[\tau_d(B)] + \mathbb{E}[\text{Var}[\text{step time} | B]]$$

$$\text{Throughput} = \mathbb{E}\left[\frac{B}{\tau_d(B)}\right] \approx \frac{\bar{B}}{\tau_d(\bar{B})} \cdot \left(1 + \frac{\sigma_B^2}{\bar{B}^2}\left(1 - \bar{B}\frac{\tau_d'(\bar{B})}{\tau_d(\bar{B})}\right)\right)$$

*Unification of previous models as special cases:*

| Limit | Reduction |
|-------|-----------|
| $K \to \infty$ (no memory limit) | Model 6 (batch-size network) |
| $\nu_{\text{preempt}} \to 0$ | Model 7 (admission control only) |
| $\text{Var} \to 0$ (deterministic) | Model 9 (mean-field fixed point) |
| $B \equiv B_0$ (fixed batch) | Model 3 (fluid queue) |
| $\tau_d(B) \equiv \tau_d$ (constant) | Model 2 (tandem queue) |
| Single-stage, $B=1$ | Model 1 (M/G/1) |

*Enforcement mechanisms:*

1. **Queueing**: Enforces work conservation via waiting
2. **Batching**: Enforces step budget conservation via batch formation
3. **Reflection**: Enforces memory conservation via boundary reflection
4. **Preemption**: Enforces hard memory limit via state jumps

*Learnable parameters (from BLIS regression):*

$$\boldsymbol{\theta} = (\tau_0, \gamma_d, \delta_d, \alpha_p, \gamma_p, \nu_0, K_{\text{thresh}}, \sigma_{\text{arrival}}, \sigma_{\text{service}})$$

Identified via:
- $\tau_0, \gamma_d, \delta_d$: Regression on step time vs. batch size
- $\alpha_p, \gamma_p$: Regression on prefill time vs. input length and batch
- $\nu_0, K_{\text{thresh}}$: Regression on preemption rate vs. memory utilization
- $\sigma$ terms: Moment matching on throughput/latency variance

**Improvement over Model 9:**
Full distributional predictions via diffusion; unifies all models; rigorous probabilistic foundation.

**Captures:**
- Mean and variance of TTFT, TPOT
- Full throughput-latency tradeoff surface
- Knee behavior (memory saturation)
- Cliff behavior (preemption cascade)
- Heavy-traffic asymptotics
- Transient dynamics via ODE/SDE integration
- All previous model behaviors as special cases

**Limitations:**
- Diffusion approximation requires heavy traffic (breaks down at very low load)
- Reflecting boundaries assume instantaneous correction (smoothing)
- Parameter fitting requires careful BLIS sampling strategy
- Computational cost: solving Fokker-Planck or simulating SDE (still much faster than discrete-event simulation)
- Sequence length distribution $\mathbf{L}$ approximated via moments, not full distribution

---

## Summary: Model Hierarchy

| Model | Key Feature | Conserved Quantity | Enforcement |
|-------|-------------|-------------------|-------------|
| 1 | Single queue | Work | FIFO queueing |
| 2 | Tandem (P→D) | Flow | Stage coupling |
| 3 | Fluid + batch | Fluid mass | Batch formation |
| 4 | State-dep. decode | Step budget | Self-consistent $\bar{B}$ |
| 5 | Continuous batch | Iteration budget | Scheduler policy |
| 6 | Service network | Population | Birth-death balance |
| 7 | KV admission | Memory (soft) | Admission control |
| 8 | Preemption | Memory (hard) | Preemption/eviction |
| 9 | Mean-field | All (ODE) | Reflection at boundary |
| 10 | Full twin | All (SDE) | Diffusion + reflection |

The final Model 10 provides an analytical twin suitable for publication, with explicit mathematical foundations, learnable parameters, and the ability to reproduce BLIS outputs orders of magnitude faster than simulation.
