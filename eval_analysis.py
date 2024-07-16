import json
import pprint as ppr
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import glob
import os
import re


def score_breakdown(eval_results_fp, model_name, save_folder=None, save_figure=False):
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
    if save_figure:
        plt.savefig(save_fp + '.png', dpi=300)
    plt.show()

    score_breakdown_results['fraction_correct'] = fraction_correct
    score_breakdown_results['problem_meta_data'] = total_in_category
    with open(save_fp + '.json', 'w') as f:
        json.dump(score_breakdown_results, f, indent=4)
    
    plt.close()


def clean_response(raw_txt):
    new_txt = '('
    valid_chars = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)] + ['?'] + [str(i) for i in range(10)]
    good_chars = []
    for sub_char in raw_txt:
        if sub_char in valid_chars:
            good_chars.append(sub_char)
    new_txt += ', '.join(good_chars) + ')'
    return new_txt


def extract_answer_better(model_answer):
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
    all_results_files = glob.glob(results_folder + results_file_prefix + '*.json')
    print(len(all_results_files))
    for fp in all_results_files:
        model_name = fp[len(results_folder + results_file_prefix):-len('.json')]
        reparse_answers(eval_results_fp=fp, 
                        model_name=model_name, 
                        overwrite=True, 
                        save_path=save_folder + results_file_prefix[:-1] + '2_' + fp[len(results_folder + results_file_prefix):]
        )


def v2_eval_runs_analysis(results_folder='./v2_results_cleaned/', save_folder='./v2_results_analysis/', results_file_prefix='rpm_eval_results2_'):
    all_results_files = glob.glob(results_folder + results_file_prefix + '*.json')
    print(len(all_results_files), all_results_files)
    for fp in all_results_files:
        model_name = fp[len(results_folder + results_file_prefix):-len('.json')]
        score_breakdown(eval_results_fp=fp, model_name=model_name, save_folder=save_folder)


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
    # score_breakdown('results/rpm_eval_results_meta-llama-Llama-3-70b-chat-hf_old.json',
    #                 'meta-llama-Llama-3-70b-chat-hf', save_folder='analysis/')
    # score_breakdown('results/rpm_eval_results_Qwen2-0.5B-Instruct.json', 'Qwen2-0.5B-Instruct')
    v2_eval_runs_corrections()
    v2_eval_runs_analysis()


if __name__ == '__main__':
    main()
