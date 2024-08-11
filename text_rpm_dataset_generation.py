import numpy as np
from typing import List, Dict, Tuple
from pprint import pprint as ppr
from text_rpm_construction import *
import random
import itertools
import tiktoken
from datetime import datetime
import json
import time
import math
from transformers import (
    AutoTokenizer
)

SUPPORTED_RULES = (
    'constant_row',
    'constant_col',
    'cycle_n',
    'cycle_n_minus_1',
    'diagonals',
)

SUPPORTED_META_RULES = (
    'general_cycle',
    'nary'
)

DEFAULT_RULES_TO_NUM_ATTRS = {
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

    @staticmethod # TODO: Problem is it's not consistent with the prompt if you use english words in the tuples etc.
    def generate_random_alphabet(values_per_attribute, num_attrs, tokenizer_path='/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct'):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                  use_fast=False,
                                                  padding_side='left',
                                                  trust_remote_code=True)
        
        tokenizer_vocab = list(tokenizer.get_vocab().keys())
        all_vocab = []
        while len(all_vocab) < values_per_attribute * num_attrs:
            sample = random.choice(tokenizer_vocab)
            try:
                sample.encode('ascii')
                all_vocab.append(sample)
            except:
                print('not unicode string, trying new')

        randomized_vocab = [all_vocab[i:i + values_per_attribute] for i in range(0, values_per_attribute * num_attrs, values_per_attribute)]
        return randomized_vocab

    @staticmethod
    def generate_default_alphabet(values_per_attribute, num_attrs, val_type='alpha'):
        if val_type == 'alpha':
            return tuple([[chr(65 + i + values_per_attribute * j) for i in range(values_per_attribute)] for j in range(int(26 / values_per_attribute))] + 
                        [[chr(97 + i + values_per_attribute * j) for i in range(values_per_attribute)] for j in range(int(26 / values_per_attribute))])
        elif val_type == 'num':
            return tuple([[i for i in range(values_per_attribute)] for j in range(num_attrs)])

    @staticmethod
    def process_meta_rules(meta_rules, num_rows, num_cols, valid_rules, rule_constructs, rules_to_num_attrs):
        assert rule_constructs is not None and rules_to_num_attrs is not None

        # nary rule type should follow unary rule types, since makes it easier when creating grid values
        if 'nary' in meta_rules:
            new_ordering = []
            for meta_rule in meta_rules:
                if meta_rule != 'nary':
                    new_ordering.append(meta_rule)
            meta_rules = new_ordering + ['nary']
        
        # Instantiate/unfold all meta rules and record their information
        for meta_rule in meta_rules:
            if meta_rule == 'general_cycle':
                new_rule_constructs = generate_all_cycle_rules(
                    n=num_rows,
                    l=num_cols,
                    attr_names=None
                )
                rule_names = list(new_rule_constructs.keys())
                valid_rules += rule_names
                rule_constructs = rule_constructs | new_rule_constructs
                rules_to_num_attrs = rules_to_num_attrs | {key: 1 for key in rule_names}
            elif meta_rule == 'general_cycle2':
                new_rule_constructs = generate_all_cycle2_rules(
                    n=num_rows,
                    l=num_cols,
                    attr_names=None
                )
                rule_names = list(new_rule_constructs.keys())
                valid_rules += rule_names
                rule_constructs = rule_constructs | new_rule_constructs
                rules_to_num_attrs = rules_to_num_attrs | {key: 1 for key in rule_names}
            elif meta_rule == 'nary':
                new_rule_constructs = generate_nary_rules(
                    n=num_rows,
                    l=num_cols,
                    a_len=num_cols,
                    b_len=num_cols,
                    nary=2,
                    attr_names=None
                )
                rule_names = list(new_rule_constructs.keys())
                valid_rules += rule_names
                rule_constructs = rule_constructs | new_rule_constructs
                rules_to_num_attrs = rules_to_num_attrs | {key: 1 for key in rule_names}
            else:
                raise ValueError(f'Meta rule {meta_rule} not recognized. Must be one of: {SUPPORTED_META_RULES}')
    
        return valid_rules, rule_constructs, rules_to_num_attrs

    @staticmethod
    def generate_rulesets(max_num_rules, rule_list, fallback_sample_size=None, max_rulesets=6**9, min_num_rules=1):
        rulesets = []
        independent_rule_list = [rule for rule in rule_list if rule[:len('nary')] != 'nary']

        # Generate all possible sequences of rules that are n long with simple recursion
        def find_all_rulesets(ruleset):
            if len(rulesets) > max_rulesets:
                return
            elif len(ruleset) == max_num_rules:
                rulesets.append(ruleset)
            else:
                for rule in rule_list:
                    if len(ruleset) > 1 or rule[:len('nary')] != 'nary':        # !! Assumes binary nary rule
                        find_all_rulesets(ruleset + [rule])
        
        # If combinatorial explosion from max num rules, just randomly sample the space
        if len(rule_list) ** max_num_rules > max_rulesets:
            # Randomly sample rulesets, making sure dependent (binary) rules are not too numerous. 1 in ~10M = 6**9 chance of duplicate (~lottery odds), so ((10M - 1)/10M)^num_samples odds of no repeats.
            assert fallback_sample_size is not None
            while len(rulesets) < fallback_sample_size:
                sampled_ruleset = random.choices(independent_rule_list, k=min(2, max_num_rules))
                if max_num_rules > 2:
                    sampled_ruleset += random.choices(rule_list, k=max_num_rules - 2)
                rulesets.append(sampled_ruleset)
        else:
            # Otherwise, brute force entire space
            find_all_rulesets([])
        
        return rulesets

    @staticmethod
    def choose_attribute_names(rules_to_num_attrs, ruleset, all_attributes):
        num_attributes = sum(rules_to_num_attrs[rule] for rule in ruleset)
        attributes = random.sample(all_attributes, k=num_attributes)
        if all_attributes[0] not in attributes:
            attr_idx = random.choice([i for i in range(len(attributes))])
            attributes[attr_idx] = all_attributes[0]
        if any('inner' in attribute for attribute in attributes) and all_attributes[1] not in attributes:
            inner_attr_idx = random.choice([i for i in range(len(attributes)) if 'inner' in attributes[i]])
            attributes[inner_attr_idx] = all_attributes[1]
        
        return attributes

    @staticmethod
    def find_valid_map_idxs(i, rule, excluded_idxs, final_ruleset): 
        all_excluded = []

        def search_exclusion_branches(rrule, rr_idx):            
            for new_excluded_idx in excluded_idxs[rrule + f'_{rr_idx}']:
                if new_excluded_idx not in all_excluded:
                    all_excluded.append(new_excluded_idx)
                    search_exclusion_branches(final_ruleset[new_excluded_idx], new_excluded_idx)

        for excluded_idx in excluded_idxs[rule + f'_{i}']:
            all_excluded.append(excluded_idx)
            search_exclusion_branches(final_ruleset[excluded_idx], excluded_idx)
        
        excluded_idxs[rule + f'_{i}'] = all_excluded
                
        return [j for j in range(len(final_ruleset)) if j not in excluded_idxs[rule + f'_{i}']], excluded_idxs

    @staticmethod
    def shuffle_ruleset_and_map(ruleset):
        independent_rules = []
        dependent_rules = []
        for rule in ruleset:
            if rule[:len('nary')] == 'nary':
                dependent_rules.append(rule)
            else:
                independent_rules.append(rule)
        
        random.shuffle(independent_rules)
        if len(dependent_rules) > 0:
            random.shuffle(dependent_rules)

            # !! Careful, assumes binary order-irrelevant binary rules !!

            insertion_idxs = random.sample(range(0, len(independent_rules) + len(dependent_rules), 1), k=len(dependent_rules))   # !! Assumes binary nary rules
            idx_0 = 0
            idx_1 = 0
            final_ruleset = []
            for i in range(len(independent_rules) + len(dependent_rules)):
                if i in insertion_idxs:
                    final_ruleset.append(dependent_rules[idx_1])
                    idx_1 += 1
                else:
                    final_ruleset.append(independent_rules[idx_0])
                    idx_0 += 1
            
            assert idx_0 == len(independent_rules) and idx_1 == len(dependent_rules) and len(independent_rules) + len(dependent_rules) == len(final_ruleset) and len(final_ruleset) == len(ruleset)

            final_maps = []
            excluded_idxs = {rule + f'_{i}': [i] if rule[:len('nary')] == 'nary' else [] for i, rule in enumerate(final_ruleset)}
            for i, rule in enumerate(final_ruleset):
                if rule[:len('nary')] == 'nary':
                    valid_idxs, excluded_idxs = RPMDataset.find_valid_map_idxs(i, rule, excluded_idxs, final_ruleset) # Have dependent rule depend on any 2 unique rules while avoiding dependency cycles

                    map_idxs = random.sample(valid_idxs, k=2) 
                    for mapped_rule, idx in zip([final_ruleset[idx] for idx in map_idxs], map_idxs):
                        excluded_idxs[mapped_rule + f'_{idx}'].append(i)
                    
                    final_maps.append([rule, map_idxs])    
                else:
                    final_maps.append([rule, []])
            
            return final_ruleset, final_maps
        else:
            return independent_rules, [[rule, []] for rule in independent_rules]

    @staticmethod
    def assign_rule_to_attribute(ruleset, attributes, rules_to_num_attrs):
        rule_to_attribute = []
        attr_idx = 0
        for rule in ruleset:
            rule_instance = [rule, attributes[attr_idx:attr_idx + rules_to_num_attrs[rule]]]
            rule_to_attribute.append(rule_instance)
            attr_idx += rules_to_num_attrs[rule]
        
        return rule_to_attribute

    @staticmethod
    def generate_rule_configs(rule_to_attribute, num_cols, attribute_to_values, fallback_sample_size=None, max_num_configs=1000, sample_fallback_threshold=10**6):
        all_configs = []
        base_num = math.factorial(len(rule_to_attribute))
        rule_to_config_space_size = {       # !! Assumes 1 attribute per rule !!
            'constant_col': base_num, 
            'constant_row': base_num, 
            'cycle_n': base_num*2, 
            'diagonals': base_num*2, 
            'cycle_n_minus_1': base_num**2, 
            'general_cycle_': base_num**2,
            'general_cycle2': base_num*len(rule_to_attribute)**len(rule_to_attribute),
            'nary': 1
        }

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
                elif rule[:len('general_cycle_')] == 'general_cycle_':
                    row_start_permutations = order_permutations
                    random.shuffle(row_start_permutations)
                    for order_permutation in order_permutations:
                        for row_start_permutation in row_start_permutations:
                            rule_config = [
                                rule, 
                                {
                                    'order': order_permutation,
                                    'row_starts': row_start_permutation,
                                }
                            ]

                            compute_config_variations(rule_idx + 1, config + [rule_config])
                elif rule[:len('general_cycle2_')] == 'general_cycle2_':
                    row_start_permutations = itertools.product([i % len(attribute_to_values[rule_to_attribute[rule_idx][1][0]]) for i in range(num_cols)], repeat=num_cols)
                    row_start_permutations = [list(item) for item in row_start_permutations]
                    random.shuffle(row_start_permutations)
                    for order_permutation in order_permutations:
                        for row_start_permutation in row_start_permutations:
                            rule_config = [
                                rule, 
                                {
                                    'order': order_permutation,
                                    'row_starts': row_start_permutation,
                                }
                            ]

                            compute_config_variations(rule_idx + 1, config + [rule_config])
                elif rule[:len('nary')] == 'nary':
                    compute_config_variations(rule_idx + 1, config + [[rule, None]])
                else:
                    raise ValueError(f'Encountered unknown rule "{rule}" while generating rule configs. This rule is not yet supported.')  

        def sample_config():
            config = []
            for rule_idx in range(len(rule_to_attribute)):
                rule = rule_to_attribute[rule_idx][0]
                order_permutations = [list(item) for item in itertools.permutations([i % len(attribute_to_values[rule_to_attribute[rule_idx][1][0]]) for i in range(num_cols)])]

                if rule in ['constant_col', 'constant_row']:
                    rule_config = [
                        rule, 
                        {
                            'order': random.choice(order_permutations),
                        }
                    ]
                elif rule in ['cycle_n', 'diagonals']:
                    sign_permutations = [-1, 1]
                    rule_config = [
                        rule, 
                        {
                            'order': random.choice(order_permutations),
                            'sign': random.choice(sign_permutations),
                        }
                    ]
                elif rule in ['cycle_n_minus_1']:
                    sign_permutations = [-1, 1]
                    shift_permutations = [0, 1, 2]
                    rule_config = [
                        rule, 
                        {
                            'order': random.choice(order_permutations),
                            'sign': random.choice(sign_permutations),
                            'shift': random.choice(shift_permutations),
                        }
                    ]
                elif rule[:len('general_cycle_')] == 'general_cycle_':
                    row_start_permutations = order_permutations
                    rule_config = [
                        rule, 
                        {
                            'order': random.choice(order_permutations),
                            'row_starts': random.choice(row_start_permutations),
                        }
                    ]
                elif rule[:len('general_cycle2')] == 'general_cycle2':
                    row_start_permutations = itertools.product([i % len(attribute_to_values[rule_to_attribute[rule_idx][1][0]]) for i in range(num_cols)], repeat=num_cols)
                    row_start_permutations = [list(item) for item in row_start_permutations]
                    rule_config = [
                        rule, 
                        {
                            'order': random.choice(order_permutations),
                            'row_starts': random.choice(row_start_permutations),
                        }
                    ]
                elif rule[:len('nary')] == 'nary':
                    rule_config = [
                        rule, None
                    ]
                else:
                    raise ValueError(f'Encountered unknown rule "{rule}" while generating rule configs. This rule is not yet supported.')
                
                config += [rule_config]
            
            return config

        # print('config space size', math.prod([rule_to_config_space_size.get(pair[0], 1) for pair in rule_to_attribute] + 
        #                                      [rule_to_config_space_size['general_cycle'] for pair in rule_to_attribute if pair[0][:len('general_cycle')] == 'general_cycle'] +
        #                                      [rule_to_config_space_size['nary'] for pair in rule_to_attribute if pair[0][:len('nary')] == 'nary']))

        # If combinatorial explosion from num rules, just randomly sample the space. 6 is a lower bound, based on only ordering config
        if math.prod([rule_to_config_space_size.get(pair[0], 1) for pair in rule_to_attribute] + 
                     [rule_to_config_space_size['general_cycle_'] for pair in rule_to_attribute if pair[0][:len('general_cycle_')] == 'general_cycle_'] +
                     [rule_to_config_space_size['general_cycle2'] for pair in rule_to_attribute if pair[0][:len('general_cycle2')] == 'general_cycle2'] +
                     [rule_to_config_space_size['nary'] for pair in rule_to_attribute if pair[0][:len('nary')] == 'nary']) > max_num_configs:
            # Randomly sample rulesets. 1 in 1M chance of duplicate (~lottery odds), so ((1M - 1)/1M)^num_samples odds of no repeats.
            assert fallback_sample_size is not None
            while len(all_configs) < fallback_sample_size:
                sampled_config = sample_config()
                if sampled_config not in all_configs:
                    all_configs.append(sampled_config)
        else:
            # Otherwise, brute force entire space        
            compute_config_variations(0, [])
        
        return all_configs

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
    def generate_and_format_problem(num_rows, 
                                    num_cols, 
                                    attributes, 
                                    attribute_to_values, 
                                    rule_to_attribute, 
                                    rule_config, 
                                    rule_constructs, 
                                    rule_maps):
        rpm_problem = RPMProblem(
            num_rows=num_rows,
            num_cols=num_cols,
            attr_names=attributes,
            attr_domains=attribute_to_values,
            rule_to_attr=rule_to_attribute,
            rule_to_ordering=rule_config,
            rule_constructs=rule_constructs,
            rule_maps=rule_maps
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
                'attribute_to_values': attribute_to_values,
                'rule_maps': rule_maps,
                'rule_config': rule_config
            }
        }

        return example
    
    @staticmethod
    def create_eval_metadata(final_problems):
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
        
        return eval_metadata

    @staticmethod
    def generate_dataset(max_num_rules, 
                         num_rows, 
                         num_cols, 
                         attribute_alphabet=None, 
                         all_attributes=DEFAULT_ATTRIBUTES, 
                         ruleset_breadth=2500, 
                         min_num_rules=1, 
                         valid_rules: List | Tuple=SUPPORTED_RULES,
                         min_configs_per_ruleset=1,
                         max_num_problems_per_num_rules=250,
                         custom_save_path=None,
                         update_interval=1000,
                         rule_constructs: Dict[str, RPMRule] | Dict=DEFAULT_RULE_CONSTRUCTS,
                         rules_to_num_attrs: Dict[str, int] | Dict=DEFAULT_RULES_TO_NUM_ATTRS,
                         meta_rules: List[str]=None,
                         default_alphabet_type: str='alpha',
                         dataset_id: str='NEW'):
        # Early check for valid custom save path name format
        if custom_save_path is not None and 'default_rpm_dataset_eval_problems_' not in custom_save_path:
            raise ValueError('If custom_save_path specified, it must incluce "default_rpm_dataset_eval_problems_".')

        # Create default alphabet
        if attribute_alphabet is None:
            attribute_alphabet = RPMDataset.generate_default_alphabet(num_cols, len(all_attributes), val_type=default_alphabet_type)
            if len(attribute_alphabet) < max_num_rules:
                raise ValueError(f'Default attribute alphabet is too small to accomodate {max_num_rules} max num rules. Please use a larger, custom alphabet (for up to {max_num_rules} attributes with {num_cols} values each) or a smaller number of max num rules.')
            
        # Create many instances of the same meta rule. Allows for more general rule instantiation.
        if meta_rules is not None:
            valid_rules, rule_constructs, rules_to_num_attrs = RPMDataset.process_meta_rules(meta_rules, num_rows, num_cols, valid_rules, rule_constructs, rules_to_num_attrs)

        print(f'Number of rules used: {len(valid_rules)}')
        print('WARNING!!!!!! USING STATIC SAVEPATH')

        # # # # Get first sample of problems
        penultimate_problems = {}

        # # Go through all possible number of rules
        for num_rules in range(min_num_rules, max_num_rules + 1, 1):
            # # Generate all possible rulesets, i.e. sequences of rules that are n long
            print('> > > NUM RULES NOW:', num_rules)
            start_time = time.time()
            all_rulesets = RPMDataset.generate_rulesets(num_rules, fallback_sample_size=max_num_problems_per_num_rules, rule_list=valid_rules, min_num_rules=min_num_rules)
            rulesets = random.sample(all_rulesets, k=min(len(all_rulesets), max_num_problems_per_num_rules))
            print('RULESETS', len(rulesets))
            
            # # Go through sequences of rules for sequence of length n
            n_rules_problems = []
            for ruleset in rulesets:
                # # Prepare problem details
                # Choose attribute names used
                attributes = RPMDataset.choose_attribute_names(rules_to_num_attrs, ruleset, all_attributes)

                # Shuffle ruleset and get rule maps (i.e. which rules each rule depends on; nary (binary) rules must be idx >= 2, but otherwise are free to stack)
                ruleset, rule_maps = RPMDataset.shuffle_ruleset_and_map(ruleset)
                random.shuffle(attributes)
                
                # Assign rules to attributes
                attribute_to_values = {attribute: values for attribute, values in zip(attributes, attribute_alphabet)}
                rule_to_attribute = RPMDataset.assign_rule_to_attribute(ruleset, attributes, rules_to_num_attrs)

                # Generate all possible rule_configs for this ruleset
                all_rule_configs = RPMDataset.generate_rule_configs(rule_to_attribute, num_cols, attribute_to_values, fallback_sample_size=min_configs_per_ruleset, max_num_configs=1000)
                rule_configs = random.sample(all_rule_configs, k=min(len(all_rule_configs), max(min_configs_per_ruleset, int((max_num_problems_per_num_rules / len(rulesets)) + 0.99))))

                # Go through all possible configs for the given ruleset
                for rule_config in rule_configs:
                    # Generate the Text RPM problem and format it for the dataset
                    example = RPMDataset.generate_and_format_problem(num_rows, 
                                                                     num_cols, 
                                                                     attributes, 
                                                                     attribute_to_values, 
                                                                     rule_to_attribute, 
                                                                     rule_config, 
                                                                     rule_constructs, 
                                                                     rule_maps)
                    n_rules_problems.append(example)
                
                    if len(n_rules_problems) % update_interval == 0:
                        print(len(n_rules_problems), 'done')
            
            penultimate_problems[num_rules] = n_rules_problems
            print(time.time() - start_time)
    
        # # # # Get second sample of problems, cutting down to final eval set
        final_problems = []
        for num_rules in penultimate_problems:
            final_problems += random.sample(penultimate_problems[num_rules], k=min(len(penultimate_problems[num_rules]), max_num_problems_per_num_rules))
                
        # # # # Save the final dataset and metadata
        path_base = 'default_rpm_dataset_eval_problems_' if custom_save_path is None or 'default_rpm_dataset_eval_problems_' not in custom_save_path else custom_save_path
        save_path = path_base + dataset_id + '.jsonl'

        with open(save_path, 'w') as f:
            for problem in final_problems:
                f.write(json.dumps(problem) + '\n')
        
        eval_metadata = RPMDataset.create_eval_metadata(final_problems)
        with open(path_base + dataset_id + '_meta_data' + '.json', 'w') as f:
            json.dump(eval_metadata, f)
        
        print('DATASET GENERATION FINISHED! FINAL DATASET DETAILS:')
        ppr(eval_metadata)


def main():
    # # Basic dataset generation example
    # RPMDataset.generate_dataset(
    #     max_num_rules=2,
    #     num_rows=3,
    #     num_cols=3
    # )

    # # General cycle dataset generation example
    # RPMDataset.generate_dataset(
    #     max_num_rules=3,
    #     num_rows=5,
    #     num_cols=5,
    #     rule_constructs={},
    #     valid_rules=[],
    #     meta_rules=['general_cycle']
    # )

    # Binary rule dataset generation example
    RPMDataset.generate_dataset(
        max_num_rules=6,
        num_rows=5,
        num_cols=5,
        rule_constructs={},
        valid_rules=[],
        meta_rules=[
            'general_cycle', 
            'general_cycle2', 
            'nary'
        ],
        default_alphabet_type='num',
        dataset_id='v6'
    )

    # # Exotic alphabet generation example
    # RPMDataset.generate_dataset(
    #     max_num_rules=5,
    #     num_rows=3,
    #     num_cols=3,
    #     attribute_alphabet=[['!', '@', '#'], ['$', '%', '^'], ['&', '-', '='], ['+', '>', '<'], ['/', '|', '~']],
    #     dataset_id='twist'
    # )


if __name__ == '__main__':
    # TODO: Add in ability to pass args via cmd line
    main()
