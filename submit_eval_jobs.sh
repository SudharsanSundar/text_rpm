#!/bin/bash

model_directories=(
    # "/data/public_models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct"  #
    "/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct"
    "/data/public_models/huggingface/meta-llama/Llama-2-13b-chat-hf"
    # "/data/public_models/huggingface/meta-llama/Llama-2-70b-chat-hf"        #
    "/data/public_models/huggingface/meta-llama/Llama-2-7b-chat-hf"
    "/data/public_models/huggingface/mistralai/Mistral-7B-Instruct-v0.3"  
    # "/data/public_models/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1"  #
    "/data/public_models/huggingface/Qwen/Qwen1.5-0.5B-Chat" 
    "/data/public_models/huggingface/Qwen/Qwen1.5-1.8B-Chat"
    "/data/public_models/huggingface/Qwen/Qwen1.5-4B-Chat"
    "/data/public_models/huggingface/Qwen/Qwen2-0.5B-Instruct"
    "/data/public_models/huggingface/Qwen/Qwen2-1.5B-Instruct"
    "/data/public_models/huggingface/Qwen/Qwen2-7B-Instruct"
    # "/data/public_models/huggingface/Qwen/Qwen2-72B-Instruct"               #
    "/data/public_models/huggingface/tiiuae/falcon-7b-instruct"
    # "/data/public_models/huggingface/tiiuae/falcon-40b-instruct"            #
    # "/data/public_models/huggingface/tiiuae/falcon-180B-chat"               #
    "/data/public_models/huggingface/deepseek-ai/deepseek-llm-7b-chat"
    # "/data/public_models/huggingface/deepseek-ai/deepseek-llm-67b-chat"     #
    "/data/public_models/huggingface/google/gemma-1.1-2b-it"
    "/data/public_models/huggingface/google/gemma-1.1-7b-it"
    "/data/public_models/huggingface/01-ai/Yi-6B-Chat"
    "/data/public_models/huggingface/01-ai/Yi-34B-Chat"
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
    batch_size=128

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
   --results_save_folder "v2_results/"

EOF

done