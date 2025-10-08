import streamlit as st
import subprocess
import pandas as pd
import plotly.express as px

# Constants for the config
CONFIG = {
    "CONTEXT_LENGTH": 7000,
    "MAX_NUM_BATCHED_TOKENS": 4096,
    "BLOCK_SIZE": 16,
    "GPU_MEM_UTIL": 0.9,
    "MODES": ["train", "test"],
    "MAX_NUM_SEQS": 4096,
    "SEED": 42,
    "GPU_TYPE": "NVIDIA-H100-80GB-HBM3",
    "GPU_MEMORY_MIN": 50000,
    "NAMESPACE": "blis",
    "CHUNK_SIZES": [256, 2048],
    "NUM_PROMPTS": 2000,
    "REQUEST_RATES": [5],
    "PREFIX_HIT_RATIOS": [0.3, 0.6],
    "DATASET_PATH": "ShareGPT_V3_unfiltered_cleaned_split.json",
    "MODEL": "Qwen/Qwen2_5-7B",  # Default model
    "SPECS": ["LL"],
    "TOTAL_KV_BLOCKS": {
        "Qwen3-14B": 14508,
        "Qwen2_5-7B": 56990
    },
    "QUEUING_DELAYS" : {
        "Qwen3-14B-LL": 5612,
        "Qwen2_5-7B-LL": 4294,
        "Qwen2_5-7B-LH": 8095
    },
    "FINISHED_DELAYS" : {
        "Qwen3-14B-LL": 582,
        "Qwen2_5-7B-LL": 518,
        "Qwen2_5-7B-LH": 935
    },
    "REGRESSION_COEFFS": {
        "Qwen3-14B": [1.17167255e-02, 1.69822525e-05, 1.86698155e-04],
        "Qwen2_5-7B": [6.59619835e-03, 8.13333252e-06, 4.11493210e-05]
    },
    "LL_SPECS": {
        "TYPE": "LL",
        "INPUT_LEN_MIN": 2,
        "INPUT_LEN_MAX": 512,
        "OUTPUT_LEN_MIN": 1,
        "OUTPUT_LEN_MAX": 10
    },
    "LH_SPECS": {
        "TYPE": "LH",
        "INPUT_LEN_MIN": 2,
        "INPUT_LEN_MAX": 512,
        "OUTPUT_LEN_MIN": 100,
        "OUTPUT_LEN_MAX": 2000
    },
    "HL_SPECS": {
        "TYPE": "HL",
        "INPUT_LEN_MIN": 512,
        "INPUT_LEN_MAX": 8000,
        "OUTPUT_LEN_MIN": 1,
        "OUTPUT_LEN_MAX": 10
    },
    "HH_SPECS": {
        "TYPE": "HH",
        "INPUT_LEN_MIN": 512,
        "INPUT_LEN_MAX": 8000,
        "OUTPUT_LEN_MIN": 100,
        "OUTPUT_LEN_MAX": 2000
    }
}

# Streamlit App Layout
st.title("Simulation Configuration")

# Model Selection
with st.container():
    st.subheader("Model Selection")
    models = ["Qwen/Qwen2.5-7B", "Qwen/Qwen3-14B"]
    selected_model = st.selectbox("Select Model", models)
    CONFIG["MODEL"] = selected_model

# Scenario Selection with multiple value selection for specifications
with st.container():
    st.subheader("Scenario Selection")
    
    # Prefix Hit Ratios - Multiple selection
    prefix_hit_ratios = st.multiselect(
        "Prefix Hit Ratios", 
        CONFIG["PREFIX_HIT_RATIOS"], 
        default=CONFIG["PREFIX_HIT_RATIOS"]  # Set default to the current values
    )
    CONFIG["PREFIX_HIT_RATIOS"] = prefix_hit_ratios
    
    # Specs Selection - Multiple selection
    specs_options = ["LL", "LH"]
    selected_specs = st.multiselect(
        "Select Specs",
        specs_options,
        default=["LL"]  # Default to "LL_SPECS"
    )
    
    # Map the selected options to the actual specs definitions in CONFIG
    selected_specs_dict = {spec: CONFIG[f"{spec}_SPECS"] for spec in selected_specs}
    CONFIG["SPECS"] = selected_specs
    
    # Show descriptions for each selected spec directly in the container
    for spec in selected_specs:
        spec_details = CONFIG[f"{spec}_SPECS"]
        st.write(f"TYPE={spec_details['TYPE']}, "
                 f"IN_MIN={spec_details['INPUT_LEN_MIN']}, "
                 f"IN_MAX={spec_details['INPUT_LEN_MAX']}, "
                 f"OUT_MIN={spec_details['OUTPUT_LEN_MIN']}, "
                 f"OUT_MAX={spec_details['OUTPUT_LEN_MAX']}")



# vLLM Config
with st.container():
    st.subheader("vLLM Config Options")
    
    # GPU Memory Utilization
    col1, col2 = st.columns([3, 4])
    with col1:
        sweep_gpu_mem = st.checkbox("GPU Memory Utilization (%)", value=True)
    with col2:
        gpu_memory_utilization = [0.9] if sweep_gpu_mem else [0.9]
        st.write(", ".join(map(str, gpu_memory_utilization)))
    CONFIG["GPU_MEM_UTIL"] = gpu_memory_utilization[0]

    # Block Size
    col1, col2 = st.columns([3, 4])
    with col1:
        sweep_block_size = st.checkbox("Block Size", value=False)
    with col2:
        block_size = [16] if not sweep_block_size else [16]
        st.write(", ".join(map(str, block_size)))
    CONFIG["BLOCK_SIZE"] = block_size[0]

    # Max Num Batched Tokens
    col1, col2 = st.columns([3, 4])
    with col1:
        sweep_max_tokens = st.checkbox("Max Num Batched Tokens", value=True)
    with col2:
        max_num_batched_tokens = [2048] if sweep_max_tokens else [2048]
        st.write(", ".join(map(str, max_num_batched_tokens)))
    CONFIG["MAX_NUM_BATCHED_TOKENS"] = max_num_batched_tokens[0]

    # Long Prefill Token Threshold
    col1, col2 = st.columns([3, 4])
    with col1:
        sweep_long_prefill = st.checkbox("Long Prefill Token Threshold", value=False)
    with col2:
        long_prefill_token_threshold = [2048] if not sweep_long_prefill else [256, 2048]
        st.write(", ".join(map(str, long_prefill_token_threshold)))
    CONFIG["CHUNK_SIZES"] = long_prefill_token_threshold

# Environment & Hardware
with st.container():
    st.subheader("Environment & Hardware")
    gpu_type = st.selectbox("GPU Type", ["H100", "A100"])
    CONFIG["GPU_TYPE"] = "NVIDIA-H100-80GB-HBM3"

# SLOs
with st.container():
    st.subheader("Goals / SLOs")
    latency_p50 = st.number_input("Latency p50 (ms)", value=40)
    latency_p95 = st.number_input("Latency p95 (ms)", value=68)
    throughput = st.number_input("Throughput (token/s)", value=5)

# Simulate Button
st.subheader("Simulation")
if st.button("Simulate w/ BLIS"):
    st.write("Running BLIS simulation...")

    # Create the experiment_constants.py file based on the selected configuration
    experiment_constants = "# Experiment Constants - Generated by Streamlit App\n\n"

    # Loop through the CONFIG dictionary to write each key-value pair
    for key, value in CONFIG.items():
        if isinstance(value, str):
            # If the value is a string, add quotes
            experiment_constants += f'{key} = "{value}"\n'
        elif isinstance(value, dict):
            # If the value is a dictionary, format it as a dictionary
            experiment_constants += f'{key} = {str(value)}\n'
        else:
            # For other types like lists or numbers, just add them directly
            experiment_constants += f'{key} = {value}\n'

    # Save to file
    with open("experiment_constants.py", "w") as f:
        f.write(experiment_constants)

    # Run the simulation script
    try:
        subprocess.run(["python", "request_rate_sweep.py"], check=True)
        # Display success message when simulation is completed
        st.success("BLIS completed, check results file")

        results_file_path = "/Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/inference-sim/results/sweep_params/simulator_results.csv"
        df = pd.read_csv(results_file_path)
        
        # Display the CSV file as a table
        st.subheader("Simulation Results")
        st.dataframe(df)  

        # Add the 'Score' column based on the SLO conditions
        df["Score"] = df.apply(
            lambda row: int(
                (row['Mean E2E(ms)'] <= latency_p50) and
                (row['P99 E2E(ms)'] <= latency_p95) and
                (row['Request throughput (req/s):'] >= throughput)
            ), axis=1
        )
        
        # Filter the rows where 'Score' is 1 (indicating SLOs are met)
        meets_slo_df = df[df["Score"] == 1]

        # Display the filtered table (rows that meet the SLOs)
        st.subheader("Rows that Meet SLOs")
        st.dataframe(meets_slo_df)  # Show the filtered DataFrame



        # Define the dimensions for the Plotly parallel categories chart
        dimensions = ["request_rate", "spec", "prefix_ratio", "chunk_size"]
        
        # Plotly Parallel Categories Chart
        fig = px.parallel_categories(
            df,
            dimensions=dimensions,
            color="Score",   # Color by the 'Score' column (0 = red, 1 = green)
            color_continuous_scale=[(0, "red"), (1, "green")],
            labels={col: col for col in df.columns}  # Label the columns nicely
        )
        
        # Display the Plotly chart
        st.subheader("Parallel Categories Chart: SLOs")
        st.plotly_chart(fig, use_container_width=True)
        
    except subprocess.CalledProcessError as e:
        st.error(f"Error running simulation: {e}")

