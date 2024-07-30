import numpy as np
from typing import List, Dict
import pprint as ppr
from text_rpm_construction import RPMProblem
import random
import itertools
import tiktoken
from datetime import datetime
import json

SUPPORTED_RULES = (
    'constant_row',
    'constant_col',
    'cycle_n',
    'cycle_n_minus_1',
    'diagonals'
)

RULES_TO_NUM_ATTRS = {
    'constant_row': 1,
    'constant_col': 1,
    'cycle_n': 1,
    'cycle_n_minus_1': 1,
    'diagonals': 1
}

DEFAULT_ATTRIBUTES = ('shape_type',     # Must be index 0. To make sure problems make basic sense
                      'inner_shape_type',       # Must be index 1. To make sure problems make basic sense
                      'shape_color',
                      'shape_size',
                      'shape_orientation',
                      'shape_texture',
                      'shape_position',
                      'shape_count',
                      'inner_shape_color',
                      'inner_shape_size',
                      'inner_shape_orientation',
                      'inner_shape_texture',
                      'inner_shape_position',
                      'inner_shape_count')

DEFAULT_PROMPT_TEMPLATE = {
    'intro': '''Consider the following pattern. Each tuple {empty_form_tuple} represents {attribute_tuple}.''',
    'row': '''Row {num}: {row}''',
    'instruction': '''\nPlease determine the correct values for the final tuple of Row {final_row}, {mystery_tuple}, which completes the pattern.'''
}


class RPMDataset:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_default_alphabet(values_per_attribute):
        return tuple([[chr(65 + i + values_per_attribute * j) for i in range(values_per_attribute)] for j in range(int(26 / values_per_attribute))] + 
                     [[chr(97 + i + values_per_attribute * j) for i in range(values_per_attribute)] for j in range(int(26 / values_per_attribute))])

    @staticmethod
    def create_prompt(attributes, problem_abstraction, num_rows, num_cols, prompt_template=DEFAULT_PROMPT_TEMPLATE):
        # Prepare the empty tuple text
        possible_empty_characters = ['_']
        assert len(set(possible_empty_characters) - set(str(problem_abstraction))) > 0
        chosen_empty_character = list(set(possible_empty_characters) - set(str(problem_abstraction)))[0]
        empty_form_values = [chosen_empty_character] * len(attributes)

        # Prepare other tuple text
        mystery_tuple = ['?'] * len(attributes)
        cleaned_attributes = [attribute.replace('_', ' ') for attribute in attributes]

        # Helper functions for formatting
        def format_row(row_vals):
            return ', '.join(['(' + ', '.join([str(val) for val in elem]) + ')' for elem in row_vals])

        def format_elem(elem):
            return '(' + ', '.join([str(val) for val in elem]) + ')'

        # Prepare the answer text and prompt text
        answer = format_elem(problem_abstraction[num_rows - 1][num_cols - 1])

        intro = prompt_template['intro'].format(empty_form_tuple=format_elem(empty_form_values),
                                                attribute_tuple=format_elem(cleaned_attributes))
        rows = '\n'.join(
            [
                prompt_template['row'].format(
                    num=i + 1,
                    row=format_row(row if i + 1 != num_rows else row[:-1] + [mystery_tuple])
                ) for i, row in enumerate(problem_abstraction)
            ]
        )
        instruction = prompt_template['instruction'].format(final_row=num_rows,
                                                            mystery_tuple=format_elem(mystery_tuple))
        
        prompt = '\n'.join([intro, rows, instruction])
        return prompt, answer

    @staticmethod
    def generate_rulesets(max_num_rules, rule_list, max_rulesets=6**12, min_num_rules=1):
        rulesets = []

        # Generate all possible sequences of rules that are n long with simple recursion
        def find_all_rulesets(ruleset):
            if len(rulesets) > max_rulesets:
                return
            elif len(ruleset) == max_num_rules:
                rulesets.append(ruleset)
            else:
                for rule in rule_list:
                    find_all_rulesets(ruleset + [rule])
        
        # If combinatorial explosion from max num rules, just randomly sample the space
        if len(rule_list) ** max_num_rules > max_rulesets:
            while len(rulesets) < max_rulesets:
                # Randomly sample rulesets. Less than 1/2 billion chance of duplicate sorted rule sequence
                while len(rulesets) < max_rulesets:
                    rulesets.append(random.choices(rule_list, k=max_num_rules))
        else:
            # Otherwise, brute force entire space
            find_all_rulesets([])
        
        return rulesets
    
    @staticmethod
    def generate_rule_configs(rule_to_attribute, num_cols, attribute_to_values, max_num_configs=1000):
        all_configs = []

        def compute_config_variations(rule_idx, config):
            # Stop searching after exhausting enough options
            if len(all_configs) >= max_num_configs:
                return
            # After assigning each rule a config, save it
            elif rule_idx == len(rule_to_attribute):
                all_configs.append(config)
            # Branch on each different config that the current rule can have
            else:
                rule = rule_to_attribute[rule_idx][0]
                order_permutations = [list(item) for item in itertools.permutations([i % len(attribute_to_values[rule_to_attribute[rule_idx][1][0]]) for i in range(num_cols)])]
                random.shuffle(order_permutations)

                if rule in ['constant_col', 'constant_row']:
                    for order_permutation in order_permutations:
                        rule_config = [
                            rule, 
                            {
                                'order': order_permutation
                            }
                        ]

                        compute_config_variations(rule_idx + 1, config + [rule_config])
                elif rule in ['cycle_n', 'diagonals']:
                    sign_permutations = [-1, 1]
                    random.shuffle(sign_permutations)
                    for order_permutation in order_permutations:
                        for sign_permutation in sign_permutations:
                            rule_config = [
                                rule, 
                                {
                                    'order': order_permutation,
                                    'sign': sign_permutation
                                }
                            ]

                            compute_config_variations(rule_idx + 1, config + [rule_config])
                elif rule in ['cycle_n_minus_1']:
                    sign_permutations = [-1, 1]
                    shift_permutations = [0, 1, 2]
                    random.shuffle(sign_permutations)
                    random.shuffle(shift_permutations)
                    for order_permutation in order_permutations:
                        for sign_permutation in sign_permutations:
                            for shift_permutation in shift_permutations:
                                rule_config = [
                                    rule, 
                                    {
                                        'order': order_permutation,
                                        'sign': sign_permutation,
                                        'shift': shift_permutation
                                    }
                                ]

                                compute_config_variations(rule_idx + 1, config + [rule_config])
                else:
                    raise ValueError(f'Encountered unknown rule "{rule}" while generating rule configs. This rule is not yet supported.')  

        compute_config_variations(0, [])
        return all_configs

    @staticmethod
    def generate_dataset(max_num_rules, 
                         num_rows, 
                         num_cols, 
                         attribute_alphabet=None, 
                         all_attributes=DEFAULT_ATTRIBUTES, 
                         ruleset_breadth=2500, 
                         min_num_rules=1, 
                         valid_rules=SUPPORTED_RULES,
                         min_configs_per_ruleset=2,
                         max_num_problems_per_num_rules=250,
                         custom_save_path=None,
                         update_interval=1000):
        if custom_save_path is not None and 'default_rpm_dataset_eval_problems_' not in custom_save_path:
            raise ValueError('If custom_save_path specified, it must incluce "default_rpm_dataset_eval_problems_".')

        if attribute_alphabet is None:
            attribute_alphabet = RPMDataset.generate_default_alphabet(num_cols)
            if len(attribute_alphabet) < max_num_rules:
                raise ValueError(f'Default attribute alphabet is too small to accomodate {max_num_rules} max num rules. Please use a larger, custom alphabet (for up to {max_num_rules} attributes with {num_cols} values each) or a smaller number of max num rules.')

        print('WARNING!!!!!! USING STATIC SAVEPATH')

        # # # # Get first sample of problems
        penultimate_problems = {}

        # Go through all possible number of rules
        for num_rules in range(min_num_rules, max_num_rules + 1, 1):
            # Generate all possible rulesets, i.e. sequences of rules that are n long
            print('> > > NUM RULES NOW:', num_rules)
            all_rulesets = RPMDataset.generate_rulesets(num_rules, rule_list=valid_rules, min_num_rules=min_num_rules)
            rulesets = random.sample(all_rulesets, k=min(len(all_rulesets), ruleset_breadth))
            print('RULESETS', len(rulesets))
            
            # Go through all possible sequences of rules used for sequence of n length
            n_rules_problems = []
            for ruleset in rulesets:
                # Prepare problem details, like attribute names used, etc.
                num_attributes = sum(RULES_TO_NUM_ATTRS[rule] for rule in ruleset)
                attributes = random.sample(all_attributes, k=num_attributes)
                if all_attributes[0] not in attributes:
                    attr_idx = random.choice([i for i in range(len(attributes))])
                    attributes[attr_idx] = all_attributes[0]
                if any('inner' in attribute for attribute in attributes) and all_attributes[1] not in attributes:
                    inner_attr_idx = random.choice([i for i in range(len(attributes)) if 'inner' in attributes[i]])
                    attributes[inner_attr_idx] = all_attributes[1]
                random.shuffle(ruleset)
                random.shuffle(attributes)
                
                attribute_to_values = {attribute: values for attribute, values in zip(attributes, attribute_alphabet)}
                rule_to_attribute = []
                attr_idx = 0
                for rule in ruleset:
                    rule_instance = [rule, attributes[attr_idx:attr_idx + RULES_TO_NUM_ATTRS[rule]]]
                    rule_to_attribute.append(rule_instance)
                    attr_idx += RULES_TO_NUM_ATTRS[rule]

                # Generate all possible rule_configs for this ruleset
                all_rule_configs = RPMDataset.generate_rule_configs(rule_to_attribute, num_cols, attribute_to_values, max_num_configs=1000)
                rule_configs = random.sample(all_rule_configs, k=min(len(all_rule_configs), max(min_configs_per_ruleset, int((min_configs_per_ruleset * ruleset_breadth) / len(all_rule_configs)))))

                # Go through all possible configs for the given ruleset
                for rule_config in rule_configs:
                    # Generate the Text RPM problem and format it for the dataset
                    rpm_problem = RPMProblem(
                        num_rows=num_rows,
                        num_cols=num_cols,
                        attr_names=attributes,
                        attr_domains=attribute_to_values,
                        rule_to_attr=rule_to_attribute,
                        rule_to_ordering=rule_config
                    )

                    problem_abstraction = rpm_problem.get_grid()
                    problem_prompt, problem_answer = RPMDataset.create_prompt(attributes, problem_abstraction, num_rows, num_cols)
                    example = {
                        'problem_prompt': problem_prompt + ' Please clearly state your final answer as "The final answer is: [your final answer]."',
                        'problem_answer': problem_answer,
                        'characteristics': {
                            'problem_abstraction': problem_abstraction,
                            'attributes': attributes,
                            'rule_to_attribute': rule_to_attribute,
                            'attribute_to_values': attribute_to_values
                        }
                    }

                    n_rules_problems.append(example)
                
                    if len(n_rules_problems) % update_interval == 0:
                        print(len(n_rules_problems), 'done')
            
            penultimate_problems[num_rules] = n_rules_problems
    
        # # # # Get second sample of problems, cutting down to final eval set
        final_problems = []
        for num_rules in penultimate_problems:
            final_problems += random.sample(penultimate_problems[num_rules], k=min(len(penultimate_problems[num_rules]), max_num_problems_per_num_rules))
        
        # Record metadata
        gpt4_tokenzier = tiktoken.encoding_for_model('gpt-4')
        eval_metadata = {
            'total_num_rules_count': {},
            'num_unique_rules_count': {},
            'total_input_tokens': {
                'tokenizer': 'tiktoken.encoding_for_model(\'gpt-4\')',
                'num_tokens': sum(len(gpt4_tokenzier.encode(problem['problem_prompt'])) for problem in final_problems)
            }
        }

        for problem in final_problems:
            num_rules = len(problem['characteristics']['rule_to_attribute'])
            num_unique_rules = len(set(pair[0] for pair in problem['characteristics']['rule_to_attribute']))

            eval_metadata['total_num_rules_count'][num_rules] = eval_metadata['total_num_rules_count'].get(num_rules, 0) + 1
            eval_metadata['num_unique_rules_count'][num_unique_rules] = eval_metadata['num_unique_rules_count'].get(num_unique_rules, 0) + 1
        
        # # # # Save the final dataset and metadata
        path_base = 'default_rpm_dataset_eval_problems_' if custom_save_path is None or 'default_rpm_dataset_eval_problems_' not in custom_save_path else custom_save_path
        # path_id = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        path_id = 'NEW'
        save_path = path_base + path_id + '.jsonl'

        with open(save_path, 'w') as f:
            for problem in final_problems:
                f.write(json.dumps(problem) + '\n')
        
        with open(path_base + 'meta_data_' + path_id + '.json', 'w') as f:
            json.dump(eval_metadata, f)
        
        print('DATASET GENERATION FINISHED! FINAL DATASET DETAILS:')
        ppr.pprint(eval_metadata)


def main():
    RPMDataset.generate_dataset(
        max_num_rules=7,
        num_rows=5,
        num_cols=5
    )


if __name__ == '__main__':
    main()
