import numpy as np
from transformers import AutoTokenizer
import json
import os
import time

class SampleRequest():
    def __init__(self, prompt, prompt_len, prefix_len, output, output_len):
        self.ID = ''
        self.input_text = prompt
        self.input_len = prompt_len
        self.generated_text = output
        self.prefix_len = prefix_len
        self.output_len = output_len
        self.e2e_latency = 0.0
        self.arrival_time = 0 # in ticks

class PrefixRepetitionRandomLengthsDataset():
    # Default values 
    DEFAULT_NUM_PREFIXES = 5
    DEFAULT_PREFIX_HIT_RATE = 10
    DEFAULT_INPUT_LEN_MEAN = 512
    DEFAULT_OUTPUT_LEN_MEAN = 200
    DEFAULT_MAX_MODEL_LEN = 8192

    def __init__(
        self,
        random_seed
    ) -> None:
        super().__init__()
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def sample(
        self,
        tokenizer,
        num_requests: int,
        input_len_mean: int,
        output_len_mean: int,
        prefix_hit_rate: int = DEFAULT_PREFIX_HIT_RATE,
        num_prefixes: int = DEFAULT_NUM_PREFIXES,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        **kwargs,
    ) -> list[SampleRequest]:
        vocab_size = tokenizer.vocab_size
        if num_requests < num_prefixes:
            raise ValueError(
                f"num_requests ({num_requests}) must be greater than or equal "
                f"to num_prefixes ({num_prefixes})"
            )

        def _generate_exact_length_tokens(target_length: int) -> list[int]:
            """Generate tokens that decode and re-encode to exactly
            target_length."""
            # Generate random tokens
            tokens = np.random.randint(
                0, vocab_size, size=target_length).tolist()
            return tokens

        requests = []
        all_prefixes = []
        for _ in range(num_prefixes):
            prefix_tokens = _generate_exact_length_tokens(100000)
            all_prefixes.append(prefix_tokens)

        for _ in range(num_requests):
            input_len = int(np.random.normal(loc=input_len_mean, scale=input_len_mean/10))
            prefix_len = int(prefix_hit_rate * input_len)
            prefix_idx = np.random.randint(0, len(all_prefixes))
            prefix_tokens = all_prefixes[prefix_idx][:prefix_len]
            suffix_len = input_len - prefix_len
            suffix_tokens = _generate_exact_length_tokens(suffix_len)

            combined_tokens = prefix_tokens + suffix_tokens
            prompt = combined_tokens
            prompt_len = len(combined_tokens)
            output_len = min(int(np.random.normal(loc=output_len_mean, scale=output_len_mean/10)), max_model_len - input_len - 10)
            output_tokens = _generate_exact_length_tokens(output_len)

            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    prefix_len=prefix_len,
                    output=output_tokens,
                    output_len=output_len
                )
            )
        np.random.shuffle(requests)
        return requests
    
def get_arrival_times(input_requests, request_rate, burstiness):
    arrival_times = []
    delay_ts = []
    for request_index, _ in enumerate(input_requests):
        if request_rate == float("inf"):
            delay_ts.append(0)
        else:
            theta = 1.0 / (request_rate * burstiness)

            # Sample the request interval from the gamma distribution.
            # If burstiness is 1, it follows exponential distribution.
            delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

    # Calculate the cumulative delay time from the first sent out requests.
    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]
    if delay_ts[-1] != 0:
        target_total_delay_s = len(input_requests) / request_rate
        normalize_factor = target_total_delay_s / delay_ts[-1]
        delay_ts = [delay * normalize_factor for delay in delay_ts]

    interval_us = 0
    for request_index, _ in enumerate(input_requests):
        arrival_times.append(interval_us)
        interval_us = int(delay_ts[request_index]*1e6)
    return arrival_times
    
def generate_synthetic_requests(seed, model, num_requests, prefix_hit_rate, input_len_mean, output_len_mean, request_rate, data_folder):
    model_name = model.split("/")[-1].replace(".", "_")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    data_dump = {"num_prompts": num_requests, "request_rate": request_rate}
    workload_generator = PrefixRepetitionRandomLengthsDataset(random_seed=seed)
    workload = workload_generator.sample(tokenizer, num_requests, input_len_mean, output_len_mean, prefix_hit_rate)
    arrival_times = get_arrival_times(workload, request_rate, burstiness=1.0)
    for idx, request in enumerate(workload):
        request.ID = f'request_{idx}'
        request.arrival_time = arrival_times[idx]
    data_model_rr_folder = os.path.join(data_folder, os.path.join(model_name, f'synthetic/rr_{request_rate}/prefix_{prefix_hit_rate}'))
    os.makedirs(data_model_rr_folder, exist_ok=True)
    data_filename = "detailed_results_inference_tokenized.json"
    data_full_filepath = os.path.join(data_model_rr_folder, data_filename)
    data_dump["prompts"] = [vars(req) for req in workload]
    with open(data_full_filepath, 'w') as f:
        json.dump(data_dump, f, indent=4)


