cd /data/sudharsan_sundar/text_rpm

python eval.py \
    --model_name_or_path "google/gemma-2-9b-it" \
    --eval_dataset_path "datasets/default_rpm_dataset_eval_problems_v2.json" \
    --results_save_folder "v2_results/" \
    --batch_size -1 \
    --use_hf_pipeline False \
    --model_org "together" \
    --use_api True