#!/bin/bash

model_directories=(     # Key: "\t#" = large model, "_##" = finished eval, "_/#" = skipping eval cuz it's annoying
    # "/data/public_models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct" ##      # running at 128 -> running again at 128 for speed
    # "/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct" ##
    # "/data/public_models/huggingface/meta-llama/Llama-2-13b-chat-hf" ##
    # "/data/public_models/huggingface/meta-llama/Llama-2-70b-chat-hf" ##       # running at 128 -> starting again cuz took over 4 hrs, 128
    # "/data/public_models/huggingface/meta-llama/Llama-2-7b-chat-hf" ##
    # "/data/public_models/huggingface/mistralai/Mistral-7B-Instruct-v0.3" ##
    # "/data/public_models/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1" ##  #
    # "/data/public_models/huggingface/Qwen/Qwen1.5-0.5B-Chat" ##
    # "/data/public_models/huggingface/Qwen/Qwen1.5-1.8B-Chat" ##
    # "/data/public_models/huggingface/Qwen/Qwen1.5-4B-Chat" ##
    # "/data/public_models/huggingface/Qwen/Qwen2-0.5B-Instruct" ##
    # "/data/public_models/huggingface/Qwen/Qwen2-1.5B-Instruct" ##
    # "/data/public_models/huggingface/Qwen/Qwen2-7B-Instruct" ##
    # "/data/public_models/huggingface/Qwen/Qwen2-72B-Instruct" ##              # ran out 128 -> running at 64
    # "/data/public_models/huggingface/tiiuae/falcon-7b-instruct" ##
    # "/data/public_models/huggingface/tiiuae/falcon-40b-instruct" #/           # !!! Careful of non-128-divisible batch size !!! running out of memory even with 64 batch size? -> even 32 by a little?? -> still crashing at 24, and it's bad 
    # "/data/public_models/huggingface/tiiuae/falcon-180B-chat" #/               # has some problem with 'token_type_ids' passed in for generate kwargs
    # "/data/public_models/huggingface/deepseek-ai/deepseek-llm-7b-chat" ##
    # "/data/public_models/huggingface/deepseek-ai/deepseek-llm-67b-chat" ##     # ran out 128 -> running at 64
    # "/data/public_models/huggingface/google/gemma-1.1-2b-it" ##
    # "/data/public_models/huggingface/google/gemma-1.1-7b-it" ##
    # "/data/public_models/huggingface/01-ai/Yi-6B-Chat" ##
    # "/data/public_models/huggingface/01-ai/Yi-34B-Chat" ##
    "/data/sudharsan_sundar/downloaded_models/gemma-2-9b-it"
)

# Initialize an empty array to hold the second halves
model_names=()

for dir in "${model_directories[@]}"; do
    model_names+=("$(basename "$dir")")
    echo "${dir}"
done

echo " "
echo " "
for i in "${!model_directories[@]}"; do
    model_path="${model_directories[i]}"
    model_name="${model_names[i]}"
    job_name="ss_${model_name}_text-rpm_eval"
    gpus_per_node=2
    batch_size=2       # Must be a power of 2 in order to make use of the "continue from previous save" feature of eval script

    echo "${model_path}"
    echo "${model_name}"
    echo "${batch_size}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=$gpus_per_node
#SBATCH --partition=cais
#SBATCH --job-name=$job_name
#SBATCH --output=/data/sudharsan_sundar/text_rpm/eval_runs_details/${model_name}/%j.out
#SBATCH --error=/data/sudharsan_sundar/text_rpm/eval_runs_details/${model_name}/%j.err

source /data/sudharsan_sundar/miniconda3/etc/profile.d/conda.sh
conda activate fluid
echo "${model_path}"
echo "${model_name}"
echo "${batch_size}"

cd /data/sudharsan_sundar/text_rpm
python eval.py \
   --model_name_or_path "${model_path}" \
   --eval_dataset_path "datasets/default_rpm_dataset_eval_problems_v2.json" \
   --batch_size $batch_size \
   --results_save_folder "v2_results/" \
   --limit_num_problems first_x,32

EOF

done