import json
import pprint as ppr
from matplotlib import pyplot as plt


def score_breakdown(eval_results_fp, model_name):
    with open(eval_results_fp, 'r') as f:
        results = []
        for item in f:
            results.append(json.loads(item))

        score_breakdown_results = {
            'total_num_rules': {},
            'num_nonconstant_rules': {},
            'num_distribute_3_rules': {},
            'num_unique_rules': {}
        }
        total_in_category = {
            'total_num_rules': {},
            'num_nonconstant_rules': {},
            'num_distribute_3_rules': {},
            'num_unique_rules': {}
        }
        num_dud_answers = 0
        num_incorrect_elems = 0

        for result in results:
            num_rules = len(result['problem_characteristics']['attribute_to_rule'].values())
            num_nonconstant_rules = len([rule for rule in result['problem_characteristics']['attribute_to_rule'].values() if rule != 'constant'])
            num_distribute_3_rules = len([rule for rule in result['problem_characteristics']['attribute_to_rule'].values() if rule == 'distribute_3'])
            num_unique_rules = len(set(result['problem_characteristics']['attribute_to_rule'].values()))

            total_in_category['total_num_rules'][num_rules] = total_in_category['total_num_rules'].get(num_rules, 0) + 1
            total_in_category['num_nonconstant_rules'][num_nonconstant_rules] = total_in_category['num_nonconstant_rules'].get(num_nonconstant_rules, 0) + 1
            total_in_category['num_distribute_3_rules'][num_nonconstant_rules] = total_in_category['num_distribute_3_rules'].get(num_distribute_3_rules, 0) + 1
            total_in_category['num_unique_rules'][num_unique_rules] = total_in_category['num_unique_rules'].get(num_unique_rules, 0) + 1

            score_breakdown_results['total_num_rules'][num_rules] = score_breakdown_results['total_num_rules'].get(num_rules, 0) + result['score']
            score_breakdown_results['num_nonconstant_rules'][num_nonconstant_rules] = score_breakdown_results['num_nonconstant_rules'].get(num_nonconstant_rules, 0) + result['score']
            score_breakdown_results['num_distribute_3_rules'][num_nonconstant_rules] = score_breakdown_results['num_distribute_3_rules'].get(num_distribute_3_rules, 0) + result['score']
            score_breakdown_results['num_unique_rules'][num_unique_rules] = score_breakdown_results['num_unique_rules'].get(num_unique_rules, 0) + result['score']

            extracted_answer = result['extracted_answer']
            if not (extracted_answer[0] == '('
                    and extracted_answer[-1] == ')'
                    and all(65 <= ord(extracted_answer[i]) <= 65 + 26 for i in range(1, len(extracted_answer), 3))      # Assumes alphabet is capital English letters
                    and all(extracted_answer[i:i + 2] == ', ' for i in range(2, len(extracted_answer) - 1, 3))):
                print('DUD answer:', extracted_answer)
                print('Correct answer:', result['correct_answer'])
                print('PROMPT')
                print(result['problem_prompt'])
                print('FULL model answer:')
                print(result['model_answer'])
                print('-'*100)
                num_dud_answers += 1
            elif len(extracted_answer.split(',')) != len(result['correct_answer'].split(',')):
                num_incorrect_elems += 1

    print('SCORE BREAKDOWN')
    print('Total answers:', len(results))
    print('Total num dud answers:', num_dud_answers)
    print('Total num incorrect elem format answers:', num_incorrect_elems)
    print('- - - Absolute total num problems:')
    ppr.pprint(total_in_category)
    print('- - - Absolute correct:')
    ppr.pprint(score_breakdown_results)
    print('- - - Fraction correct:')
    fraction_correct = {category: {num: score_breakdown_results[category][num] / total_in_category[category][num] for num in total_in_category[category]} for category in total_in_category}
    ppr.pprint(fraction_correct)

    fig, axes = plt.subplots(2, 2, figsize=(12, 5))

    for i, category in enumerate(fraction_correct):
        ax = axes[i // 2, i % 2]
        keys = list(fraction_correct[category].keys())
        values = list(fraction_correct[category].values())

        ax.bar(keys, values, color='skyblue')
        ax.set_xlabel(' '.join(category.split('_')))
        ax.set_ylabel('Fraction of problems correct')
        ax.set_title(category)

    title = f'custom RPM eval results for {model_name}'
    fig.suptitle(title)
    plt.tight_layout()

    save_fp = '_'.join(title.replace('/', '-').split(' ')) + '.png'
    plt.savefig(save_fp, dpi=300)
    plt.show()


def main():
    score_breakdown('rpm_eval_results_meta-llama-Llama-3-8b-hf.json', 'meta-llama-Llama-3-8b-hf')


if __name__ == '__main__':
    main()

