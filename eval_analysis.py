import json
import pprint as ppr
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import glob
import os
import re
import math

evaled_model_fps = [
    "/data/public_models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct", ##      # running at 128 -> running again at 128 for speed
    "/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct", ##
    "/data/public_models/huggingface/meta-llama/Llama-2-13b-chat-hf", ##
    "/data/public_models/huggingface/meta-llama/Llama-2-70b-chat-hf", ##       # running at 128 -> starting again cuz took over 4 hrs, 128
    "/data/public_models/huggingface/meta-llama/Llama-2-7b-chat-hf", ##
    "/data/public_models/huggingface/mistralai/Mistral-7B-Instruct-v0.3", ##
    "/data/public_models/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1", ##  #
    "/data/public_models/huggingface/Qwen/Qwen1.5-0.5B-Chat", ##
    "/data/public_models/huggingface/Qwen/Qwen1.5-1.8B-Chat", ##
    "/data/public_models/huggingface/Qwen/Qwen1.5-4B-Chat", ##
    "/data/public_models/huggingface/Qwen/Qwen2-0.5B-Instruct", ##
    "/data/public_models/huggingface/Qwen/Qwen2-1.5B-Instruct", ##
    "/data/public_models/huggingface/Qwen/Qwen2-7B-Instruct", ##
    "/data/public_models/huggingface/Qwen/Qwen2-72B-Instruct", ##              # ran out 128 -> running at 64
    "/data/public_models/huggingface/tiiuae/falcon-7b-instruct", ##
    # "/data/public_models/huggingface/tiiuae/falcon-40b-instruct", #/           # !!! Careful of non-128-divisible batch size !!! running out of memory even with 64 batch size? -> even 32 by a little?? -> still crashing at 24, and it's bad 
    # "/data/public_models/huggingface/tiiuae/falcon-180B-chat", #/               # has some problem with 'token_type_ids' passed in for generate kwargs
    "/data/public_models/huggingface/deepseek-ai/deepseek-llm-7b-chat", ##
    "/data/public_models/huggingface/deepseek-ai/deepseek-llm-67b-chat", ##     # ran out 128 -> running at 64
    "/data/public_models/huggingface/google/gemma-1.1-2b-it", ##
    "/data/public_models/huggingface/google/gemma-1.1-7b-it", ##
    "/data/public_models/huggingface/01-ai/Yi-6B-Chat", ##
    "/data/public_models/huggingface/01-ai/Yi-34B-Chat", ##
    "/data/sudharsan_sundar/downloaded_models/gemma-2-9b-it"
]

evaled_models = [fp.split('/')[-1] for fp in evaled_model_fps]

model_to_param_count = {
  "Meta-Llama-3-70B-Instruct": "70B",
  "Meta-Llama-3-8B-Instruct": "8B",
  "Llama-2-13b-chat-hf": "13B",
  "Llama-2-70b-chat-hf": "70B",
  "Llama-2-7b-chat-hf": "7B",
  "Mistral-7B-Instruct-v0.3": "7B",
  "Mixtral-8x7B-Instruct-v0.1": "8x7B",
  "Qwen1.5-0.5B-Chat": "0.5B",
  "Qwen1.5-1.8B-Chat": "1.8B",
  "Qwen1.5-4B-Chat": "4B",
  "Qwen2-0.5B-Instruct": "0.5B",
  "Qwen2-1.5B-Instruct": "1.5B",
  "Qwen2-7B-Instruct": "7B",
  "Qwen2-72B-Instruct": "72B",
  "falcon-7b-instruct": "7B",
  "falcon-40b-instruct": "40B",
  "falcon-180B-chat": "180B",
  "deepseek-llm-7b-chat": "7B",
  "deepseek-llm-67b-chat": "67B",
  "gemma-1.1-2b-it": "2B",
  "gemma-1.1-7b-it": "7B",
  "Yi-6B-Chat": "6B",
  "Yi-34B-Chat": "34B"
}
model_to_param_count = {key: float(value[:-1]) for key, value in zip(model_to_param_count.keys(), model_to_param_count.values()) if key != 'Mixtral-8x7B-Instruct-v0.1'}
model_to_param_count['Mixtral-8x7B-Instruct-v0.1'] = 45

model_to_tokens_seen = {
  "Meta-Llama-3-70B-Instruct": 15_000_000_000_000,
  "Meta-Llama-3-8B-Instruct": 15_000_000_000_000,
  "Llama-2-13b-chat-hf": 2_000_000_000_000,
  "Llama-2-70b-chat-hf": 2_000_000_000_000,
  "Llama-2-7b-chat-hf": 2_000_000_000_000,
#   "Mistral-7B-Instruct-v0.3": null,
#   "Mixtral-8x7B-Instruct-v0.1": null,
#   "Qwen1.5-0.5B-Chat": null,
#   "Qwen1.5-1.8B-Chat": null,
#   "Qwen1.5-4B-Chat": null,
#   "Qwen2-0.5B-Instruct": null,
#   "Qwen2-1.5B-Instruct": 7_200_000_000_000,       # "roughly over 7T", https://github.com/QwenLM/Qwen2/issues/515
#   "Qwen2-7B-Instruct": 7_200_000_000_000,
#   "Qwen2-72B-Instruct": 7_200_000_000_000,
  "falcon-7b-instruct": 1_500_000_000_000,
  "falcon-40b-instruct": 1_000_000_000_000,
  "falcon-180B-chat": 3_500_000_000_000,
  "deepseek-llm-7b-chat": 2_000_000_000_000,
  "deepseek-llm-67b-chat": 2_000_000_000_000,
  "gemma-1.1-2b-it": 3_000_000_000_000,
  "gemma-1.1-7b-it": 6_000_000_000_000,
  "Yi-6B-Chat": 3_100_000_000_000,
  "Yi-34B-Chat": 3_100_000_000_000
}

model_to_gscore = {
    'Qwen1.5-0.5B-Chat': -7.60,
    'Qwen1.5-1.8B-Chat': -4.54,
    'gemma-1.1-2b-it': -4.07,
    'falcon-7b-instruct': -3.68,
    'Qwen1.5-4B-Chat': -3.44,
    'Qwen1.5-7B-Chat': -2.34,
    'Llama-2-7b-chat-hf': -1.85,
    'gemma-1.1-7b-it': -1.12,
    'Llama-2-13b-chat-hf': -0.76,
    'Yi-6B-Chat': -0.67,
    'Qwen1.5-14B-Chat': -0.66,
    'deepseek-llm-7b-chat': -0.62,
    'falcon-40b-instruct': 0.56,
    'Mistral-7B-Instruct-v0.2': 0.73,
    'Qwen1.5-72B-Chat': 0.74,
    'Qwen1.5-32B-Chat': 0.94,
    'Meta-Llama-3-8B-Instruct': 1.13,
    'Llama-2-70b-chat-hf': 1.21,
    'Yi-34B-Chat': 1.71,
    'Qwen1.5-110B-Chat': 2.43,
    'falcon-180B-chat': 2.58,
    'deepseek-llm-67b-chat': 2.91,
    'Mixtral-8x7B-Instruct-v0.1': 3.36,
    'dbrx-instruct': 3.60,
    'Meta-Llama-3-70B-Instruct': 4.57,
    'Mixtral-8x22B-Instruct-v0.1': 4.87
}

model_to_values = {
    "Meta-Llama-3-70B-Instruct": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 82,
        'human_eval': 81.7,
        'gsm8k': 93,
        'math': 50.4,
        'gpqa': 39.5
    },
    "Meta-Llama-3-8B-Instruct": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 68.4,
        'human_eval': 62.2,
        'gsm8k': 79.6,
        'math': 30,
        'gpqa': 34.2
    },
    "Llama-2-13b-chat-hf": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 54.8,
        'gpqa': None,
        'human_eval': 18.3,
        'gsm8k': 28.7,
        'math': None,
    },
    "Llama-2-70b-chat-hf": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 68.9,
        'gpqa': None,
        'human_eval': 29.9,
        'gsm8k': 56.8,
        'math': None,
    },
    "Llama-2-7b-chat-hf": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 45.3,
        'gpqa': None,
        'human_eval': 12.8,
        'gsm8k': 14.6,
        'math': None,
    },
    "Mistral-7B-Instruct-v0.3": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': None,
        'human_eval': None,
        'gsm8k': None,
        'math': None,
    },
    "Mixtral-8x7B-Instruct-v0.1": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': None,
        'gpqa': None,
        'human_eval': None,
        'gsm8k': None,
        'math': None,
    },
    "Qwen1.5-0.5B-Chat": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': None,
        'gpqa': None,
        'human_eval': None,
        'gsm8k': None,
        'math': None,
    },
    "Qwen1.5-1.8B-Chat": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': None,
        'gpqa': None,
        'human_eval': None,
        'gsm8k': None,
        'math': None,
    },
    "Qwen1.5-4B-Chat": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': None,
        'gpqa': None,
        'human_eval': None,
        'gsm8k': None,
        'math': None,
    },
    "Qwen2-0.5B-Instruct": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': None,
        'gpqa': None,
        'human_eval': None,
        'gsm8k': None,
        'math': None,
    },
    "Qwen2-1.5B-Instruct": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': None,
        'gpqa': None,
        'human_eval': None,
        'gsm8k': None,
        'math': None,
    },
    "Qwen2-7B-Instruct": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 70.5,
        'gpqa': None,
        'human_eval': 79.9,
        'gsm8k': 82.3,
        'math': 49.6,
    },
    "Qwen2-72B-Instruct": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 82.3,
        'gpqa': 42.4,
        'human_eval': 86,
        'gsm8k': 91.1,
        'math': 59.7,
    },
    "falcon-7b-instruct": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 28,       # Not sure if this is instruct
        'gpqa': None,
        'human_eval': None,
        'gsm8k': None,
        'math': None,
    },
    "falcon-40b-instruct": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 57,
        'gpqa': None,
        'human_eval': None,
        'gsm8k': None,
        'math': None,
    },
    "falcon-180B-chat": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 70.6,
        'gpqa': None,
        'human_eval': 35.4,
        'gsm8k': None,
        'math': None,
    },
    "deepseek-llm-7b-chat": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 49.4,
        'gpqa': None,
        'human_eval': 48.2,
        'gsm8k': 63,
        'math': 15.8,
    },
    "deepseek-llm-67b-chat": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 71.1,
        'gpqa': None,
        'human_eval': 73.8,
        'gsm8k': 84.1,
        'math': 32.6,
    },
    "gemma-1.1-2b-it": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 42.3,       # Note sure if this is instruct
        'gpqa': None,
        'human_eval': 22,
        'gsm8k': 17.7,
        'math': 11.8,
    },
    "gemma-1.1-7b-it": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 64.3,
        'gpqa': None,
        'human_eval': 32.3,
        'gsm8k': 46.4,
        'math': 24.3,
    },
    "Yi-6B-Chat": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 60.99,  # taking their 5-shot, since this is common
        'gpqa': None,
        'human_eval': None,
        'gsm8k': 44.88,     # taking 4-shot, arbitrarily
        'math': None,
    },
    "Yi-34B-Chat": {
        'num_params': None,
        'tokens_seen': None,
        'mmlu': 73.46,
        'gpqa': None,
        'human_eval': None,
        'gsm8k': 75.97,
        'math': None,
    },
}
print(set(model_to_param_count.keys()) - set(model_to_values.keys()))
for key in model_to_param_count.keys():
    model_to_values[key]['num_params'] = model_to_param_count[key]
    if key in model_to_tokens_seen:
        model_to_values[key]['tokens_seen'] = model_to_tokens_seen[key]
    if key in model_to_gscore:
        model_to_values[key]['gscore'] = model_to_gscore[key]
    else:
        model_to_values[key]['gscore'] = None


def score_breakdown(eval_results_fp, model_name, save_folder=None, save_figure=False):
    '''
    Analyze how models perform on various subsets of the text RPM eval results
    '''
    with open(eval_results_fp, 'r') as f:
        results = []
        for item in f:
            results.append(json.loads(item))

        score_breakdown_results = {
            'total_num_rules': {},
            'num_nonconstant_rules': {},
            'num_distribute_3_rules': {},
            'num_unique_rules': {},
            'num_unparsable_answers': 0,
            'num_parsed_but_incorrectly_formatted_answers': 0,
            'num_formatted_but_improper_answer': 0,
            'overall_accuracy': 0,
        }
        total_in_category = {
            'total_num_rules': {},
            'num_nonconstant_rules': {},
            'num_distribute_3_rules': {},
            'num_unique_rules': {}
        }

        for result in results:
            num_rules = len(result['problem_characteristics']['attribute_to_rule'].values())
            num_nonconstant_rules = len(
                [rule for rule in result['problem_characteristics']['attribute_to_rule'].values() if
                 rule != 'constant'])
            num_distribute_3_rules = len(
                [rule for rule in result['problem_characteristics']['attribute_to_rule'].values() if
                 rule == 'distribute_3'])
            num_unique_rules = len(set(result['problem_characteristics']['attribute_to_rule'].values()))

            total_in_category['total_num_rules'][num_rules] = total_in_category['total_num_rules'].get(num_rules, 0) + 1
            total_in_category['num_nonconstant_rules'][num_nonconstant_rules] = total_in_category[
                                                                                    'num_nonconstant_rules'].get(
                num_nonconstant_rules, 0) + 1
            total_in_category['num_distribute_3_rules'][num_nonconstant_rules] = total_in_category[
                                                                                     'num_distribute_3_rules'].get(
                num_distribute_3_rules, 0) + 1
            total_in_category['num_unique_rules'][num_unique_rules] = total_in_category['num_unique_rules'].get(
                num_unique_rules, 0) + 1

            score_breakdown_results['total_num_rules'][num_rules] = score_breakdown_results['total_num_rules'].get(
                num_rules, 0) + result['score']
            score_breakdown_results['num_nonconstant_rules'][num_nonconstant_rules] = score_breakdown_results[
                                                                                          'num_nonconstant_rules'].get(
                num_nonconstant_rules, 0) + result['score']
            score_breakdown_results['num_distribute_3_rules'][num_nonconstant_rules] = score_breakdown_results[     # TODO: something is wrong here. everything is num noncons 0
                                                                                           'num_distribute_3_rules'].get(
                num_distribute_3_rules, 0) + result['score']
            score_breakdown_results['num_unique_rules'][num_unique_rules] = score_breakdown_results[
                                                                                'num_unique_rules'].get(
                num_unique_rules, 0) + result['score']

            extracted_answer = result['extracted_answer']
            if extracted_answer == 'Could not parse answer.':
                score_breakdown_results['num_unparsable_answers'] += 1
            elif not (extracted_answer[0] == '('
                      and extracted_answer[-1] == ')'
                      and all((65 <= ord(extracted_answer[i]) <= 97 + 26 or 65 <= ord(extracted_answer[i]) <= 97 + 26 or extracted_answer[i] == '?' or extracted_answer[i] in [i for i in range(10)]) for i in range(1, len(extracted_answer), 3))
                      and all(extracted_answer[i:i + 2] == ', ' for i in range(2, len(extracted_answer) - 1, 3))):
                # print('DUD answer:', extracted_answer)
                # print('Correct answer:', result['correct_answer'])
                # print('PROMPT')
                # print(result['problem_prompt'])
                # print('FULL model answer:')
                # print(result['model_answer'])
                # print('-' * 100)

                score_breakdown_results['num_parsed_but_incorrectly_formatted_answers'] += 1
            elif len(extracted_answer.split(',')) != len(result['correct_answer'].split(',')) or not (all((65 <= ord(extracted_answer[i]) <= 97 + 26 or 65 <= ord(extracted_answer[i]) <= 97 + 26) for i in range(1, len(extracted_answer), 3))):
                score_breakdown_results['num_formatted_but_improper_answer'] += 1

    score_breakdown_results['overall_accuracy'] = sum(
        score_breakdown_results['total_num_rules'][num_rules] for num_rules in
        score_breakdown_results['total_num_rules']) / len(results)
    fraction_correct = {
        category: {num: score_breakdown_results[category][num] / total_in_category[category][num] for num in
                   total_in_category[category]} for category in total_in_category}

    print('SCORE BREAKDOWN,', model_name)
    print('Total answers:', len(results))
    print('Overall accuracy:', score_breakdown_results['overall_accuracy'])
    print('Total num unparsable answers:', score_breakdown_results['num_unparsable_answers'])
    print('Total num incorrectly formatted answers:',
          score_breakdown_results['num_parsed_but_incorrectly_formatted_answers'])
    print('Total num improper (shape, values, etc.) answers:', score_breakdown_results['num_formatted_but_improper_answer'])
    if model_name in []:
        print('- - - Absolute total num problems:')
        ppr.pprint(total_in_category)
        print('- - - Absolute correct:')
        ppr.pprint(score_breakdown_results)
        print('- - - Fraction correct:')
        ppr.pprint(fraction_correct)
    
    title = f'rpm eval analysis/{model_name}'
    save_fp = '_'.join(title.replace('/', '-').split(' '))
    if save_folder is not None:
        save_fp = save_folder + save_fp
    
    score_breakdown_results['fraction_correct'] = fraction_correct
    score_breakdown_results['problem_meta_data'] = total_in_category
    with open(save_fp + '.json', 'w') as f:
        json.dump(score_breakdown_results, f, indent=4)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    del fraction_correct['num_nonconstant_rules']
    del fraction_correct['num_distribute_3_rules']

    for i, category in enumerate(fraction_correct):
        ax = axes[i] # axes[i // 2, i % 2]
        keys = list(fraction_correct[category].keys())
        values = list(fraction_correct[category].values())

        bars = ax.bar(keys, values, color='skyblue')
        ax.set_xlabel(' '.join(category.split('_')))
        ax.set_ylabel('Fraction of problems correct')
        ax.set_title(category)

        for bar, key in zip(bars, keys):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'n={total_in_category[category][key]}', ha='center',
                    va='bottom')

    fig.suptitle(title)
    plt.tight_layout()

    if save_figure:
        plt.savefig(save_fp + '.png', dpi=300)
    plt.show()
    
    plt.close()


def clean_response(raw_txt):
    '''
    Model answer parsing utility
    '''
    new_txt = '('
    valid_chars = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)] + ['?'] + [str(i) for i in range(10)]
    good_chars = []
    for sub_char in raw_txt:
        if sub_char in valid_chars:
            good_chars.append(sub_char)
    new_txt += ', '.join(good_chars) + ')'
    return new_txt


def extract_answer_better(model_answer):
    '''
    Another model answer parsing utility
    '''
    if model_answer.lower().rfind('the final answer is: ') != -1:
        first_cut = model_answer[model_answer.rfind('final answer is: ') + len('final answer is: '):]
        second_cut = first_cut.split(')')[0]
        extracted_answer = second_cut + ')'

        return extracted_answer
    else:
        backup_pattern = r'\([A-Za-z, ]+\)'
        backup_find = list(re.findall(backup_pattern, model_answer))
        if len(backup_find) > 0:
            return backup_find[-1]
        else:
            return 'Could not parse answer.'


def reparse_answers(eval_results_fp, model_name, overwrite=False, save_path=None):
    '''
    Go through eval results and reparse answers using better answer extraction logic/methods.

    Initial answer parsing methods incorrectly parsed model answers a not insignificant number of times.
    So I did some error analysis/manual investigation of examples to do more thorough parsing.
    '''
    results = []
    with open(eval_results_fp, 'r') as f:
        for item in f:
            results.append(json.loads(item))
        
    print('> '*10, 'DOING:', model_name)
    invalid_response_words = [
        'missing',
        'unknown',
        'unanswered',
        'lack',
        'further',
        'cannot',
        'more',
        'none',
        'valid',
        'sequence',
        'pattern',
        'letters',
        'additional',
        'placeholder',
        'shape',
        'correct',
        'blank',
        'true',
        'information',
        'specific',
        'determine',
        'your final answer',
        'relationship',
        'context',
        'the final tuple of row 3',
        'no final answer',
        'the number of',
        'no value provided',
        'triangle',
        'star',
        'circle',
        'diamond',
        'square',
        'blue',
        'green',
        'element',
        'orange',
        'red'
    ]

    counts = [0, 0, 0, 0, 0, 0, 0]
    total_change = 0
    idxs = []
    i = 0

    new_results = []
    for old_result in results:
        result = old_result.copy()
        extracted_answer = result['extracted_answer']
        bracket_extract = '' if extracted_answer.find('[') == -1 or extracted_answer.find(']') == -1 else extracted_answer[extracted_answer.find('['):extracted_answer.find(']') + 1]        

        # Skip parse errors: TODO: done?
        if extracted_answer == 'Could not parse answer.':
            counts[1] += 1

            new_extract = extract_answer_better(result['model_answer'])
            if new_extract != 'Could not parse answer.' and not(any(word in new_extract.lower() for word in invalid_response_words)):
                new_extract = clean_response(new_extract)
                result['score'] = int(new_extract == result['correct_answer'])
                result['extracted_answer'] = new_extract
                result['old_extract'] = extracted_answer
                total_change += abs(int(new_extract == result["correct_answer"]) - int(extracted_answer == result["correct_answer"]))
        # Skip refusal answers
        elif any(word in extracted_answer.lower() for word in invalid_response_words) and (len(bracket_extract) > 0 and any(word in bracket_extract.lower() for word in invalid_response_words) or len(bracket_extract) == 0):
            counts[0] += 1
            
            if 'the final tuple of row 3 is' in extracted_answer.lower():
                new_extract = extracted_answer[len('the final tuple in row 3 is '):] if ':' not in extracted_answer else extracted_answer[len('the final tuple in row 3 is: '):]
                result['score'] = int(new_extract == result['correct_answer'])
                result['extracted_answer'] = new_extract
                result['old_extract'] = extracted_answer
                total_change += abs(int(new_extract == result["correct_answer"]) - int(extracted_answer == result["correct_answer"]))
            elif extracted_answer[2] == '\n':
                new_extract = f'({extracted_answer[0]})'
                result['score'] = int(new_extract == result['correct_answer'])
                result['extracted_answer'] = new_extract
                result['old_extract'] = extracted_answer
                total_change += abs(int(new_extract == result["correct_answer"]) - int(extracted_answer == result["correct_answer"]))
        # Catch answers that aren't in proper [ABCabc?] tuple format
        elif not (extracted_answer[0] == '('
                  and extracted_answer[-1] == ')'
                  and all((65 <= ord(extracted_answer[i]) <= 97 + 26 or 65 <= ord(extracted_answer[i]) <= 97 + 26 or extracted_answer[i] == '?' or extracted_answer[i] in [i for i in range(10)]) for i in range(1, len(extracted_answer), 3))  # Assumes alphabet is capital and lowercase English letters and single digits
                  and all(extracted_answer[i:i + 2] == ', ' for i in range(2, len(extracted_answer) - 1, 3))):
            counts[2] += 1
            cleaned_answer = None

            # If answer is bracketed, clean bracketed portion
            if len(bracket_extract) > 0:
                counts[3] += 1
                cleaned_answer = clean_response(bracket_extract)
            # Other common cases
            elif 'the final tuple in row 3 is' in extracted_answer.lower():
                cleaned_answer = extracted_answer[len('the final tuple in row 3 is '):] if ':' not in extracted_answer else extracted_answer[len('the final tuple in row 3 is: '):]
            elif 'the final tuple for row 3 is' in extracted_answer.lower():
                cleaned_answer = extracted_answer[len('the final tuple for row 3 is '):] if ':' not in extracted_answer else extracted_answer[len('the final tuple for row 3 is: '):]
            elif 'the final answer is ' in extracted_answer.lower():
                cleaned_answer = extracted_answer[len('the final answer is '):] if ':' not in extracted_answer else extracted_answer[len('the final answer is: '):]
                cleaned_answer = clean_response(cleaned_answer)
            elif '[your final answer]" would be: ' in extracted_answer.lower():
                cleaned_answer = extracted_answer[len('[your final answer]" would be: '):]
                cleaned_answer = clean_response(cleaned_answer)
            # If answer is not bracketed (often parens), clean whole answer
            else:
                counts[4] += 1
                cleaned_answer = clean_response(extracted_answer)
                # print('CLEANED:', extracted_answer, '->', cleaned_answer)
            
            result['score'] = int(cleaned_answer == result['correct_answer'])
            result['extracted_answer'] = cleaned_answer
            result['old_extract'] = extracted_answer

            total_change += abs(int(cleaned_answer == result["correct_answer"]) - int(extracted_answer == result["correct_answer"]))
            idxs.append(i)
        
        new_results.append(result)
        i += 1
    
    # Check amount of changes and catches
    print(['could not parse', 'refusal/lang', 'non-perfect', 'brackets', 'parens'])
    print(counts)
    print(total_change)

    # Save if desired
    if overwrite:
        assert save_path is not None
        with open(save_path, 'w') as f:
            for result in new_results:
                f.write(json.dumps(result) + '\n')


def v2_eval_runs_corrections(results_folder='./v2_results/', save_folder='./v2_results_cleaned/', results_file_prefix='rpm_eval_results_'):
    '''
    Run answer reparsing on all eval results, and save to different dir
    '''
    all_results_files = glob.glob(results_folder + results_file_prefix + '*.json')
    print(len(all_results_files))
    for fp in all_results_files:
        model_name = fp[len(results_folder + results_file_prefix):-len('.json')]
        reparse_answers(eval_results_fp=fp, 
                        model_name=model_name, 
                        overwrite=True, 
                        save_path=save_folder + results_file_prefix[:-1] + '2_' + fp[len(results_folder + results_file_prefix):]
        )


def v2_eval_runs_analysis(results_folder='./v2_results_cleaned/', 
                          save_folder='./v2_results_analysis/', 
                          results_file_prefix='rpm_eval_results2_', 
                          save_figure=False):
    '''
    Run score breakdown for all eval results
    '''
    all_results_files = glob.glob(results_folder + results_file_prefix + '*.json')
    print(len(all_results_files), all_results_files)
    for fp in all_results_files:
        if True:
            model_name = fp[len(results_folder + results_file_prefix):-len('.json')]
            score_breakdown(eval_results_fp=fp, model_name=model_name, save_folder=save_folder, save_figure=save_figure)


def find_capabilities_correlations(models=evaled_models, 
                                   analysis_fp='./v2_results_analysis/rpm_eval_analysis-{model_name}.json',
                                   mode='all',
                                   comparison_data='num_params',
                                   save_folder='./v2_capabilities_comparisons/'):
    '''
    Plot model performance on text rpm tasks against various capabilities-related measures for the same models, and find corresponding correlation coefficients
    '''
    model_to_textrpm_acc = {}
    for model in models:
        with open(analysis_fp.format(model_name=model), 'r') as f:
            results = json.load(f)
            if mode == 'all':
                model_to_textrpm_acc[model] = results['overall_accuracy']
            else:
                model_to_textrpm_acc[model] = sum(results['total_num_rules'][str(num)] for num in mode) / sum(results['problem_meta_data']['total_num_rules'][str(num)] for num in mode)
    
    comparison_model_to_values = {}
    for model in models:
        if model_to_values[model][comparison_data] is not None:
            comparison_model_to_values[model] = model_to_values[model][comparison_data]
    
    # Then align on keys, and plot them against one another
    plot_models = list(set.intersection(set(models), set(comparison_model_to_values.keys())))
    print(len(models))
    print(len(plot_models))
    textrpm_vals = [model_to_textrpm_acc[model] for model in plot_models]
    textrpm_vals = [math.log(val) for val in textrpm_vals]
    comparison_vals = [comparison_model_to_values[model] for model in plot_models]
    print(len(textrpm_vals))
    print(len(comparison_vals))

    plt.scatter(comparison_vals, textrpm_vals, color='blue')
    for i, label in enumerate(plot_models):
        plt.text(comparison_vals[i], textrpm_vals[i], label, fontsize=6, ha='right')
    plt.title(f'Variation of TextRPM performance over {comparison_data}')
    plt.xlabel(comparison_data)
    plt.ylabel(f'text_rpm_acc (num_rules={mode}) (logged)')

    plt.savefig(f'{save_folder}{comparison_data}_vs_textrpm_acc_log.png', dpi=300)

    r2 = r2_score(comparison_vals, textrpm_vals)
    corr = np.corrcoef(comparison_vals, textrpm_vals)[0, 1]
    corr2, _ = pearsonr(comparison_vals, textrpm_vals)
    corr3, pval = spearmanr(a=comparison_vals, b=textrpm_vals)
    print(f'R^2 between {comparison_data} and text RPM acc (LOG): {r2} (skl), {corr} (np) | Pearson corr coeff {corr2} | spearman {corr3} / {pval}')

    plt.show()
    plt.close()


def main():
    # v2_eval_runs_corrections()
    # v2_eval_runs_analysis(save_figure=True)
    # comparison_options = ['num_params', 'tokens_seen', 'mmlu', 'human_eval', 'gsm8k', 'math', 'gscore']
    comparison_options = ['gscore']
    mode = 'all'
    # mode = [13, 14]
    for opt in comparison_options:
        find_capabilities_correlations(comparison_data=opt, mode=mode)


if __name__ == '__main__':
    main()
