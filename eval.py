from dataset import RPMDataset
from model import APIModel, ClusterModel
from transformers import set_seed
import random
import json
import re
import tiktoken
import time
import argparse
import pprint as ppr

gpt4_tokenzier = tiktoken.encoding_for_model('gpt-4')
random.seed(42)
set_seed(42)


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


def eval_model_on_rpm_batched(model_name_or_path, 
                              eval_dataset_path, 
                              results_save_folder, 
                              limit_num_problems=None, 
                              api=True, 
                              stop_seqs=None, 
                              batch_size=2, 
                              model_org=None,
                              use_hf_pipeline=False):
    if api:
        raise NotImplementedError('Haven\'t gotten batched eval set up for API models yet.')
        assert not(api and use_hf_pipeline), 'Use of API (hosted model) is mututally exclusive with use of HF pipeline (local model).'
    else:
        model = ClusterModel(model_name_or_path=model_name_or_path, batch_size=batch_size)

    with open(eval_dataset_path, 'r') as f:
        eval_problems = []
        for item in f:
            eval_problems.append(json.loads(item))

    if limit_num_problems is not None:
        print(f'NOTE: Taking {limit_num_problems} to use out of all given eval problems.')
        if limit_num_problems['method'] == 'sample':
            eval_problems = random.sample(eval_problems, k=limit_num_problems['num_problems'])
        elif limit_num_problems['method'] == 'first_x':
            eval_problems = eval_problems[:limit_num_problems['num_problems']]
        else:
            raise ValueError('Currenrtly, the only valid methods for limiting number of problems is \'sample\' (randomly sample from all problems) and \'first_x\' (take the first x problems, often roughly the x easiest problems)')

    results = []
    total_score = 0
    start_time = time.time()

    base_path = '/data/sudharsan_sundar/text_rpm/'
    save_path = 'rpm_eval_results_' + model.model_name.replace('/', '-') + '.json'
    if results_save_folder is not None:
        save_path = results_save_folder + save_path
    save_path = base_path + save_path
    # try:
    #     with open(save_path, 'r') as f:
    #         print('WARNING: Already some file at save location. Overwriting...')
    #     with open(save_path, 'w') as f:
    #         print('Overwritten.')
    # except:
    #     print('No existing save file detecting. Will save to new file.')
    
    if use_hf_pipeline:
        model_answers = model.get_answer_text_batched(eval_problems)

        for problem, model_answer in zip(eval_problems, model_answers):
            prompt = problem['problem_prompt']
            correct_answer = problem['problem_answer']
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
            print('EXTRACTED ANSWER | CORRECT ANSWER')
            print(extracted_answer, '|', result['correct_answer'])
            print('SCORE | TOTAL SCORE')
            print(result['score'], '|', total_score, '/', len(results))
            print('-'*100)
    else:
        for i in range(0, len(eval_problems), batch_size):
            new_eval_problems = eval_problems[i:min(i + batch_size, len(eval_problems))]
            new_model_answers = model.get_answer_text_batched_alt([new_eval_problem['problem_prompt'] for new_eval_problem in new_eval_problems])
            new_results = []

            for problem, model_answer in zip(new_eval_problems, new_model_answers):
                prompt = problem['problem_prompt']
                correct_answer = problem['problem_answer']
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
                new_results.append(result)
                total_score += result['score']

                print('PROMPT')
                print(prompt)
                print('MODEL ANSWER')
                print(model_answer)
                print('EXTRACTED ANSWER | CORRECT ANSWER')
                print(extracted_answer, '|', result['correct_answer'])
                print('SCORE | TOTAL SCORE')
                print(result['score'], '|', total_score, '/', len(results))
                print('-'*100)

            with open(save_path, 'a') as f:
                for result in new_results:
                    f.write(json.dumps(result) + '\n')

    with open(save_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    totals = {'total_score': total_score, 'fraction_correct': total_score / len(results), 'total_time_mins': (time.time() - start_time) / 60}
    save_path = 'rpm_eval_totals_' + model.model_name.replace('/', '-') + '.json'
    if results_save_folder is not None:
        save_path = results_save_folder + save_path
    with open(save_path, 'w') as f:
        json.dump(totals, f, indent=4)

    print(f'TOTAL SCORE: {total_score} / {len(eval_problems)} problems correct ({total_score / len(eval_problems)})')
    print(f'TOTAL TIME TAKEN FOR EVAL: {(time.time() - start_time) / 60} min ({(time.time() - start_time) / (60 * len(eval_problems))} min on avg per problem)')


def main():     # scancel ss__text-rpm_eval
    parser = argparse.ArgumentParser(description='Evaluate models on text RPM problems using batched inputs.')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Name of model or path to model to evaluate.')
    parser.add_argument('--eval_dataset_path', type=str, required=True, help='Path to eval problems from text rpm dataset.')
    parser.add_argument('--results_save_folder', type=str, required=True, help='Path to folder to save results in.')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size to use when doing model inference.')
    parser.add_argument('--limit_num_problems', type=str, default=None, help='Whether to use a subset of problems for testing. Format as \'method,num_problems\', such as \'first_x,100\'.')
    parser.add_argument('--use_hf_pipeline', type=bool, default=False, help='Whether to use the HF pipeline class to conduct inference, or use a custom implementation.')
    parser.add_argument('--use_api', type=bool, default=False, help='Whether to use a model API. Not yet implemented.')
    args = parser.parse_args()

    eval_model_on_rpm_batched(model_name_or_path=args.model_name_or_path,
                              eval_dataset_path=args.eval_dataset_path,
                              results_save_folder=args.results_save_folder,
                              batch_size=args.batch_size,
                              limit_num_problems={'method': args.limit_num_problems.split(',')[0], 'num_problems': int(args.limit_num_problems.split(',')[1])},
                              use_hf_pipeline=args.use_hf_pipeline,
                              api=args.use_api)
    
    # TODO: 
    
    # CLUSTER TESTING
    # eval_model_on_rpm_batched(model_name='Meta-Llama-3-8B-Instruct',
    #                           api=False,
    #                           eval_dataset_path='default_rpm_dataset_eval_problems_7-8.json',
    #                           results_save_folder='results/',
    #                           limit_num_problems={'method': 'sample', 'num_problems': 20},
    #                           batch_size=10,
    #                           use_hf_pipeline=False)

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
    # eval_model_on_rpm(model_name='meta-llama/Llama-3-70b-chat-hf',
    #                   model_org='together',
    #                   api=True,
    #                   eval_dataset_path='default_rpm_dataset_eval_problems_7-8.json',
    #                   results_save_folder='results/',
    #                   limit_num_problems=None)
    

if __name__ == '__main__':
    main()
