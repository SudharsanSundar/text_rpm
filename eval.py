from dataset import RPMDataset
from model import APIModel, ClusterModel
import random
import json
import re
import tiktoken
import time

gpt4_tokenzier = tiktoken.encoding_for_model('gpt-4')
random.seed(42)


def extract_answer(model_answer):
    if model_answer.lower().rfind('the final answer is: ') != -1:
        first_cut = model_answer[model_answer.rfind('final answer is: ') + len('final answer is: '):]
        second_cut = first_cut.split(')')[0]
        extracted_answer = second_cut + ')'

        return extracted_answer
    else:
        backup_pattern = r'\([A-Z, ]+\)'     # WARNING: assumes values are capital letters A-Z
        backup_find = list(re.findall(backup_pattern, model_answer))
        if len(backup_find) > 0:
            return backup_find[-1]
        else:
            return 'Could not parse answer.'


def eval_model_on_rpm(model_name, model_org, eval_dataset_path, results_save_folder, limit_num_problems=None, api=True, stop_seqs=None):
    if api:
        model = APIModel(model_name=model_name, org=model_org)
    else:
        model = ClusterModel(model_name=model_name)

    with open(eval_dataset_path, 'r') as f:
        eval_problems = []
        for item in f:
            eval_problems.append(json.loads(item))

    if limit_num_problems is not None:
        print(f'NOTE: Taking {limit_num_problems} to use out of all given eval problems.')
        if limit_num_problems['method'] == 'sample':
            eval_problems = random.sample(eval_problems, k=limit_num_problems['num_problems'])
        if limit_num_problems['method'] == 'first_x':
            eval_problems = eval_problems[:limit_num_problems['num_problems']]

    results = []
    total_score = 0
    start_time = time.time()
    for problem in eval_problems:
        prompt = problem['problem_prompt']
        correct_answer = problem['problem_answer']
        model_answer = model.get_answer_text(prompt=prompt, stop_seqs=stop_seqs)
        extracted_answer = extract_answer(model_answer)

        result = {
            'problem_prompt': prompt,
            'correct_answer': correct_answer,
            'model_answer': model_answer,
            'extracted_answer': extracted_answer,
            'score': int(extracted_answer == correct_answer),
            'problem_characteristics': problem['characteristics']
        }

        results.append(result)
        total_score += result['score']

        print('PROMPT')
        print(prompt)
        print('MODEL ANSWER')
        print(model_answer)
        print('EXTRACTED ANSWER')
        print(extracted_answer)
        print('CORRECT ANSWER')
        print(result['correct_answer'])
        print('SCORE')
        print(result['score'])
        print('-'*100)

    save_path = 'rpm_eval_results_' + model_name.replace('/', '-') + '.json'
    if results_save_folder is not None:
        save_path = results_save_folder + save_path

    with open(save_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f'TOTAL SCORE: {total_score} / {len(eval_problems)} problems correct ({total_score / len(eval_problems)})')
    print(f'TOTAL TIME TAKEN FOR EVAL: {(time.time() - start_time) / 60} min ({(time.time() - start_time) / (60 * len(eval_problems))} min on avg per problem)')


def main():
    # Size ~1.5B
    # # Raw | ~30? / 1000 (easy) problems (~0.03 min per problem); 38 / 1000 = 3.8% correct
    # eval_model_on_rpm(model_name='Qwen/Qwen1.5-1.8B',
    #                   model_org='together',
    #                   api=True,
    #                   stop_seqs=['''system'''],
    #                   eval_dataset_path='default_rpm_dataset_eval_problems_7-8.json',
    #                   results_save_folder='results/',
    #                   limit_num_problems={'method': 'first_x', 'num_problems': 1000})
    # # # Chat | ~25 min / 1000 (easy) problems (~0.025 min per problem); 23 / 1000 = 2.3% correct
    # eval_model_on_rpm(model_name='Qwen/Qwen1.5-1.8B-Chat',      # Answers are significantly dumber than base model, based on looking at a few initial answers
    #                   model_org='together',
    #                   api=True,
    #                   eval_dataset_path='default_rpm_dataset_eval_problems_7-8.json',
    #                   results_save_folder='results/',
    #                   limit_num_problems={'method': 'first_x', 'num_problems': 1000})

    # # Size ~7B
    # # # Mistral
    # eval_model_on_rpm(model_name='mistralai/Mistral-7B-Instruct-v0.3',
    #                   model_org='together',
    #                   api=True,
    #                   eval_dataset_path='default_rpm_dataset_eval_problems_7-8.json',
    #                   results_save_folder='results/',
    #                   limit_num_problems=None)
    #
    # # # LLaMA
    # # # # Raw
    # eval_model_on_rpm(model_name='meta-llama/Llama-3-8b-hf',
    #                   model_org='together',
    #                   api=True,
    #                   eval_dataset_path='default_rpm_dataset_eval_problems_7-8.json',
    #                   results_save_folder='results/',
    #                   limit_num_problems=None)
    # # # # Chat
    # # # # # Full eval
    # eval_model_on_rpm(model_name='meta-llama/Llama-3-8b-chat-hf',
    #                   model_org='together',
    #                   api=True,
    #                   eval_dataset_path='default_rpm_dataset_eval_problems_7-8.json',
    #                   results_save_folder='results/',
    #                   limit_num_problems=None)
    # # # # Chat
    # # # # # Easy subset
    # eval_model_on_rpm(model_name='meta-llama/Llama-3-8b-chat-hf',
    #                   model_org='together',
    #                   api=True,
    #                   eval_dataset_path='default_rpm_dataset_eval_problems_7-8.json',
    #                   results_save_folder='results/',
    #                   limit_num_problems={'method': 'first_x', 'num_problems': 1000})
    #
    # Size ~70B
    # # LLaMA
    # # # Raw / Wasn't answering question. Need few-shot prompt in order to do this.
    # eval_model_on_rpm(model_name='meta-llama/Meta-Llama-3-70B',
    #                   model_org='together',
    #                   api=True,
    #                   eval_dataset_path='rpm_eval_dataset_eval_problems_old.json',
    #                   results_save_folder='results/',
    #                   limit_num_problems=None)
    # # # # Chat | ~354.6 min / 8942 problems (~0.04 min per problem); 6554 / 8942 = 73.29% correct
    eval_model_on_rpm(model_name='meta-llama/Llama-3-70b-chat-hf',
                      model_org='together',
                      api=True,
                      eval_dataset_path='default_rpm_dataset_eval_problems_7-8.json',
                      results_save_folder='results/',
                      limit_num_problems=None)


if __name__ == '__main__':
    main()
