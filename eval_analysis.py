import json
import pprint as ppr
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr


def score_breakdown(eval_results_fp, model_name, save_folder=None):
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
            'num_formatted_but_incorrect_shape_answer': 0,
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
            score_breakdown_results['num_distribute_3_rules'][num_nonconstant_rules] = score_breakdown_results[
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
                      and all(65 <= ord(extracted_answer[i]) <= 65 + 26 for i in
                              range(1, len(extracted_answer), 3))  # Assumes alphabet is capital English letters
                      and all(extracted_answer[i:i + 2] == ', ' for i in range(2, len(extracted_answer) - 1, 3))):
                print('DUD answer:', extracted_answer)
                print('Correct answer:', result['correct_answer'])
                print('PROMPT')
                print(result['problem_prompt'])
                print('FULL model answer:')
                print(result['model_answer'])
                print('-' * 100)
                score_breakdown_results['num_parsed_but_incorrectly_formatted_answers'] += 1
            elif len(extracted_answer.split(',')) != len(result['correct_answer'].split(',')):
                score_breakdown_results['num_formatted_but_incorrect_shape_answer'] += 1

    score_breakdown_results['overall_accuracy'] = sum(
        score_breakdown_results['total_num_rules'][num_rules] for num_rules in
        score_breakdown_results['total_num_rules']) / len(results)

    print('SCORE BREAKDOWN')
    print('Total answers:', len(results))
    print('Overall accuracy:', score_breakdown_results['overall_accuracy'])
    print('Total num unparsable answers:', score_breakdown_results['num_unparsable_answers'])
    print('Total num incorrectly formatted answers:',
          score_breakdown_results['num_parsed_but_incorrectly_formatted_answers'])
    print('Total num incorrectly shaped answers:', score_breakdown_results['num_formatted_but_incorrect_shape_answer'])
    print('- - - Absolute total num problems:')
    ppr.pprint(total_in_category)
    print('- - - Absolute correct:')
    ppr.pprint(score_breakdown_results)
    print('- - - Fraction correct:')
    fraction_correct = {
        category: {num: score_breakdown_results[category][num] / total_in_category[category][num] for num in
                   total_in_category[category]} for category in total_in_category}
    ppr.pprint(fraction_correct)

    fig, axes = plt.subplots(2, 2, figsize=(12, 5))

    for i, category in enumerate(fraction_correct):
        ax = axes[i // 2, i % 2]
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

    title = f'rpm eval analysis/{model_name}'
    fig.suptitle(title)
    plt.tight_layout()

    save_fp = '_'.join(title.replace('/', '-').split(' '))
    if save_folder is not None:
        save_fp = save_folder + save_fp
    plt.savefig(save_fp + '.png', dpi=300)
    plt.show()

    score_breakdown_results['fraction_correct'] = fraction_correct
    score_breakdown_results['problem_meta_data'] = total_in_category
    with open(save_fp + '.json', 'w') as f:
        json.dump(score_breakdown_results, f)


def correlation_calculator():
    model_keys = ['Qwen1.5-1.8B-Chat', 'Llama3-8B-Chat', 'Llama3-70B-Chat']
    text_rpm = {
        'Qwen1.5-1.8B-Chat': 0.02,  # 1000 easy question subset
        'Llama3-8B-Chat': 0.07827216628775577,  # old sampling of old dataset
        'Llama3-70B-Chat': 0.7329456497427869  # new sampling of old dataset
    }
    mmlu = {
        'Qwen1.5-1.8B-Chat': 0.468,
        'Llama3-8B-Chat': 0.684,
        'Llama3-70B-Chat': 0.82
    }
    math = {
        'Qwen1.5-1.8B-Chat': 0.101,
        'Llama3-8B-Chat': 0.3,
        'Llama3-70B-Chat': 0.504
    }
    human_eval = {
        'Qwen1.5-1.8B-Chat': 0.201,
        'Llama3-8B-Chat': 0.622,
        'Llama3-70B-Chat': 0.817
    }
    lm_sys_arena = {
        'Qwen1.5-1.8B-Chat': None,
        'Llama3-8B-Chat': 1157,
        'Llama3-70B-Chat': 1207
    }
    arc_agi = {
        'Qwen1.5-1.8B-Chat': None,
        'Llama3-8B-Chat': None,
        'Llama3-70B-Chat': None
    }

    benchmark_scores = {
        'text_rpm': text_rpm,
        'mmlu': mmlu,
        'math': math,
        'human_eval': human_eval,
        # 'lm_sys_arena': lm_sys_arena,
    }

    def plot_capabilities_correlation(dictX, dictY, keys, titleX, titleY):
        for key in keys:
            plt.scatter([dictX[key]], [dictY[key]], label=key)

        plt.xlabel(titleX)
        plt.ylabel(titleY)
        plt.title('Correlation between Capabilities Measures')

        x_data = [dictX[key] for key in keys]
        y_data = [dictY[key] for key in keys]
        r2 = r2_score(x_data, y_data)

        corr = np.corrcoef(x_data, y_data)[0, 1]

        corr2, _ = pearsonr(x_data, y_data)
        corr3, pval = spearmanr(a=x_data, b=y_data)
        print(f'R^2 between {measure1} and {measure2}: {corr}, {corr2}, {corr3} / {pval}')

        plt.legend()
        plt.text(0.5, 0.5, f'$pearson = {corr:.2f}$')
        plt.show()

    for measure1 in benchmark_scores:
        for measure2 in benchmark_scores:
            if measure1 != measure2 and 'text_rpm' in [measure1, measure2]:
                plot_capabilities_correlation(benchmark_scores[measure1], benchmark_scores[measure2], model_keys, measure1, measure2)


def main():
    # correlation_calculator()
    # score_breakdown('results/rpm_eval_results_meta-llama-Llama-3-8b-chat-hf_old.json', 'meta-llama/Llama-3-8b-chat-hf', save_folder='analysis/')
    # score_breakdown('results/rpm_eval_results_Qwen-Qwen1.5-1.8B.json', 'Qwen/Qwen1.5-1.8B')
    # score_breakdown('results/rpm_eval_results_Qwen-Qwen1.5-1.8B-Chat.json', 'Qwen/Qwen1.5-1.8B-Chat')
    score_breakdown('results/rpm_eval_results_meta-llama-Llama-3-70b-chat-hf_old.json',
                    'meta-llama-Llama-3-70b-chat-hf', save_folder='analysis/')


if __name__ == '__main__':
    main()
