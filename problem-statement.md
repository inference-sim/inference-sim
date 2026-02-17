Problem statement:
- objective: currently blis contains two modeling techniques for inference performance:
  - blackbox optimization approach as documented in @docs/approach.md and
  - roofline approach in @docs/roofline.md
- come up with a third approach that can simulate diverse settings such as
  - model type/architecture: i.e. dense vs. MoE
  - different workload types: i.e. prefill- and decode-heavy and mixed/balanced workloads
  - different hardware: i.e. A100, H100
  - different tensor parallelism sizes and expert parallel settings
  - different vLLM knobs: i.e. chunk size, max-model-len, and --cpu-offloading
- constraints
  - We still want alpha (used to compute the delay between request arrival and queuing) and beta (used to compute the vLLM busy-loop step time) coefficients, but you have freedom to determine what alpha and beta coefficients need to be to achieve objective.
  - We can heavily featurize each setting. You can derive any new features using a model's config.json, the hardware specs (will be provided through data sheets in JSON), vLLM configuration specs, and request characteristics. These are known for each simulation.
  - carefully look into the request journey tracing, step tracing, and KV event streams documented in @vllm.md. Make sure the coefficient vectors alpha and beta can be learned using the tracing and KV event stream data. Provide a short description of the training pipeline. It can include anything from simple linear regression to advanced techniques like expectation maximization, convex optimization, or anything else that is relevant
  - The arrival to queuing latency is alpha * feature_vec_1 and the step-time latency is beta * feature_vec_2 (the `*` represents dot product). Feel free to derive the features in any way you think is appropriate. Show your reasoning and explain why the features meet the constraints and objectives.
  - we want the training procedure to not overfit but be robust

Create a collection of subagents with the following cumulative exercise:

0. @docs/plans/inference-modeling-ideas-2.md contains all the background.
1. idea creator: comes up with new proposals that solve the problem. Append the idea to a running document @docs/plans/inference-modeling-ideas-2.md. Make the idea brief.
2. three judges, each that will execute the /review-plan skill against the newly generated idea in latest iteration. Copy @docs/plans/inference-modeling-ideas-2.md as context. Use a different model for idea evaluation. You can select from `aws/claude-opus-4-6`, `Azure/gpt-4o`, `GCP/gemini-2.5-flash`. YOU HAVE TO USE THE /review-plan SKILL AND INVOKE THE TWO NON-ANTHROPIC MODELS. The judges will provide independently review feedback.
3. a summarizer appends the review summaries to a running document. The intent is that we will generate a single idea per `n` iteration.
4. In each iteration, the idea creator will run first, followed by the three judges in parallel, followed by the summarizer. In any iteration, the idea creator can use all the feedback previously given available in the document.
5. The document contains the proposal and review summary for all the iterations finished so far.

Run ten iterations and then stop.