# TODO: set up eval on models. Start with small llama models, work up to bigger models. I think they'll trounce this though, seems very easy.
# - ideally make it easy to get score cross sections, such as score on things with x rule, score with x number non constant rules, with x rules total, etc.

from dataset import RPMDataset
from model import APIModel
import random
import json
import re
import tiktoken

gpt4_tokenzier = tiktoken.encoding_for_model('gpt-4')


def create_eval_problem_set(rpm_dataset, og_path, problem_config=None):
    if problem_config is None:
        problem_config = {
            0: 0.5,
            1: 0.5,
            2: 0.3,
            3: 0.3,
            4: 0.1,
            5: 0.1,
        }

    eval_problems = []
    total_tokens = 0
    for num_rules in rpm_dataset.full_dataset:
        rule_segment = rpm_dataset.full_dataset[num_rules]

        for rule_instance in rule_segment:
            attributes = rule_instance['attributes']
            attribute_to_rule = rule_instance['attribute_to_rule']
            attribute_to_values = rule_instance['attribute_to_values']
            num_nonconstant_rules = rule_instance['num_nonconstant_rules']

            if random.random() > problem_config[num_nonconstant_rules]:
                for problem_prompt, problem_answer, problem_abstraction in zip(rule_instance['problem_prompts'], rule_instance['problem_answers'], rule_instance['problem_abstractions']):
                    if random.random() > 0.5:
                        eval_problem = {
                            'problem_prompt': problem_prompt + ' Please clearly state your final answer as "The final answer is: [your final answer]."',
                            'problem_answer': problem_answer,
                            'characteristics': {
                                'problem_abstraction': problem_abstraction,
                                'attributes': attributes,
                                'attribute_to_rule': attribute_to_rule,
                                'attribute_to_values': attribute_to_values,
                                'num_nonconstant_rules': num_nonconstant_rules
                            }
                        }
                        eval_problems.append(eval_problem)
                        total_tokens += len(gpt4_tokenzier.encode(problem_prompt))

    print(f'TOTAL EVAL PROBLEMS: {len(eval_problems)}')
    print(f'TOTAL INPUT TOKENS: {total_tokens}')
    with open(og_path[:-5] + '_eval_problems.json', 'w') as f:
        for problem in eval_problems:
            f.write(json.dumps(problem) + '\n')

    return eval_problems


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


def eval_model_on_rpm(model_name, model_org, eval_dataset_path, results_save_path, problem_config, limit_num_problems=1000):
    api_model = APIModel(model_name=model_name, org=model_org)

    if 'eval_problems' not in eval_dataset_path:
        rpm_dataset = RPMDataset()
        rpm_dataset.load_dataset(eval_dataset_path)
        eval_problems = create_eval_problem_set(rpm_dataset, eval_dataset_path, problem_config)
    else:
        with open(eval_dataset_path, 'r') as f:
            eval_problems = []
            for item in f:
                eval_problems.append(json.loads(item))

    if limit_num_problems is not None:
        eval_problems = random.sample(eval_problems, k=limit_num_problems)

    results = []
    total_score = 0
    for problem in eval_problems:
        prompt = problem['problem_prompt']
        correct_answer = problem['problem_answer']
        model_answer = api_model.get_answer_text(prompt=prompt)

        print('PROMPT')
        print(prompt)
        print('MODEL ANSWER')
        print(model_answer)
        extracted_answer = extract_answer(model_answer)
        print('EXTRACTED ANSWER')
        print(extracted_answer)

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

        print('CORRECT ANSWER')
        print(result['correct_answer'])
        print('SCORE')
        print(result['score'])
        print('-'*100)

    with open(results_save_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f'TOTAL SCORE: {total_score} / {len(eval_problems)} problems correct ({total_score / len(eval_problems)})')


def main():
    eval_model_on_rpm('meta-llama/Llama-3-8b-hf', 'together', 'rpm_eval_dataset_eval_problems.json', 'rpm_eval_results.json', None, limit_num_problems=None)


if __name__ == '__main__':
    main()
