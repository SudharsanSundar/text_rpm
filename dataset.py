from rpm import RPMMaker
import random
import pprint as ppr
import json
import tiktoken
from datetime import datetime

DEFAULT_ALPHABET = tuple([[chr(65 + i + 3 * j) for i in range(3)] for j in range(8)])
DEFAULT_ATTRIBUTES = ('shape_type',
                      'inner_shape_type',
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
SUPPORTED_RULES = ('constant', 'progression', 'distribute_3')
DEFAULT_SAMPLING = {'base_num_exs': 10000,
                    '0_num_noncons_rules_prob': 0.1,
                    '1_num_noncons_rules_prob': 0.1,
                    '2_num_noncons_rules_prob': 0.3,
                    '3_num_noncons_rules_prob': 0.3,
                    '4_num_noncons_rules_prob': 0.3,
                    '5_num_noncons_rules_prob': 0.3,
                    '6_num_noncons_rules_prob': 0.3,
                    'unspecified_num_noncons_rules_prob': 0.3}  # Just explicitly stating what's happening in the code, this value isn't directly used


class RPMDataset:
    """
    RPMDataset structure (after generation/loading):

    Actual dataset is in self.full_dataset

    self.full_dataset =
    {
        '1_num_rules': [        # This is a specific combination of n rules.
            {
                'attributes': list[str]     # What attributes of objects are mentioned, e.g. shape type, shape color, inner shape number, etc.
                'attribute_to_rule': dict[str: str]      # What pattern is used to vary/determine the value of each attribute for each cell of a row
                'attribute_to_values': dict[str: list[str]]     # The 'alphabet' for each attribute, i.e. all the values that attribute can take on
                'problem_abstractions': [
                    [['A', 'A', 'A'], ['B', 'B', 'B'], ['C', 'C', 'C']],    # The full grid instance used in the problem, represented by a list[list[str]] 3x3 2d list
                    .
                    .
                    .
                ]
                'problem_prompts': [
                    'Consider the following pattern...',
                    'Consider the following pattern...',
                    .
                    .
                    .
                ]
                'problem_answers': [
                    '(A)',
                    '(B)',
                    '(A, C)',
                    .
                    .
                    .
                ]
                'num_problems': int
                'num_nonconstant_rules': int    # number of rules used that are not 'constant', i.e. all those that are 'progression' or 'distribute_3'
            },
            {...},
            .
            .
            .
        ]
        '2_num_rules': ...
        .
        .
        .
    }

    self.eval_dataset = [
        {
            'problem_prompt': str,
            'problem_answer': str,
            'characteristics': {
                'problem_abstraction': list[list[str or int]],
                'attributes': list[str],
                'attribute_to_rule': dict[str: str],
                'attribute_to_values': dict[str: list[str or int]],
                'num_nonconstant_rules': int
            }
        }
    ]
    """
    def __init__(self,
                 min_rules=1,
                 max_rules=6,
                #  max_constant_adds=3,
                 alphabet=DEFAULT_ALPHABET,
                 all_attributes=DEFAULT_ATTRIBUTES,
                 ruleset=SUPPORTED_RULES,
                 sampling_scheme=None):
        self.min_num_rules = min_rules
        self.max_num_rules = max_rules
        # self.max_constant_additions = max_constant_adds     # TODO: Not used rn. Also, don't think this would change much.

        self.alphabet = alphabet
        assert 'type' in all_attributes[0]
        assert 'inner' in all_attributes[1] and 'type' in all_attributes[1]
        self.all_attributes = all_attributes
        self.rules = ruleset
        self.sampling_scheme = sampling_scheme if sampling_scheme is not None else DEFAULT_SAMPLING

        self.full_dataset = None
        self.len_dataset = None
        self.eval_dataset = None
        self.eval_dataset_len = None
        self.eval_metadata = None

        self.storage_path = None
        self.eval_storage_path = None

    def generate_dataset(self):
        final_problems = {}
        total_num_problems = 0
        rpmmaker = RPMMaker()
        for num_rules in range(self.min_num_rules, self.max_num_rules + 1, 1):
            print(self.alphabet, 'NUM RULES NOW', num_rules)

            # Generate all possible combinations of rules that are a sequence of num_rules long
            all_rule_combos = []
            self.generate_rule_combos(num_rules, 0, None, all_rule_combos)

            # Loop through each possible rule combination and create problem instantiations based on the given rule combo
            all_iteration_outputs = []
            for rule_combo in all_rule_combos:
                # Generate all info for problem specifications
                attributes = [self.all_attributes[0]] + random.sample(self.all_attributes, k=num_rules-1)
                random.shuffle(attributes)
                random.shuffle(rule_combo)
                if any('inner' in attribute for attribute in attributes) and self.all_attributes[1] not in attributes:
                    inner_attr_idx = random.choice([i for i in range(len(attributes)) if 'inner' in attributes[i]])
                    attributes[inner_attr_idx] = self.all_attributes[1]

                attribute_to_rule = {attribute: rule for attribute, rule in zip(attributes, rule_combo)}
                attribute_to_values = {attribute: values for attribute, values in zip(attributes, self.alphabet)}
                
                attributes_out, all_problems = rpmmaker.generate_all_unique_problems_from_rules(attribute_to_rule, attribute_to_values)     # 10k search max. Consider making randomization better.

                # Take some subset of these problem instances. Current scheme tries include maximum diversity of rule_combos, taking very small numbers from each combo.
                num_nonconstant_rules = sum(1 for rule in attribute_to_rule.values() if rule != 'constant')
                num_problems_used = min(len(all_problems), max(4, int((self.sampling_scheme['base_num_exs'] * self.sampling_scheme.get(f'{num_nonconstant_rules}_num_noncons_rules_prob', 0.3) + 1) / len(all_rule_combos))))
                print('num used', num_problems_used)
                all_problems_used = random.sample(all_problems, k=num_problems_used)

                # Prepare sampled problems to be added to dataset
                added_problems = {
                    'attributes': attributes_out,
                    'attribute_to_rule': attribute_to_rule,
                    'attribute_to_values': attribute_to_values,
                    'problem_abstractions': all_problems_used,
                    'problem_prompts': [rpmmaker.make_prompt(attributes_out, problem)[0] for problem in all_problems_used],
                    'problem_answers': [rpmmaker.make_prompt(attributes_out, problem)[1] for problem in all_problems_used],
                    'num_problems': len(all_problems_used),
                    'num_nonconstant_rules': num_nonconstant_rules
                }
                all_iteration_outputs.append(added_problems)

                total_num_problems += len(all_problems_used)
                print('Total problems so far', total_num_problems)

            # Add all sampled problems from this round of generation to the dataset
            final_problems[f'{str(num_rules)}_num_rules'] = all_iteration_outputs

        self.full_dataset = final_problems
        self.len_dataset = total_num_problems
        self.save_dataset('default_rpm_dataset.json')

    def generate_rule_combos(self, max_num_rules, idx, combo, all_combos):
        if idx == max_num_rules:
            all_combos.append(combo)
        else:
            for rule in self.rules:
                if combo is None:
                    combo = []
                self.generate_rule_combos(max_num_rules, idx + 1, combo + [rule], all_combos)

    def save_dataset(self, save_path):
        self.storage_path = save_path
        to_save = {'full_dataset': self.full_dataset, 'len_dataset': self.len_dataset}
        with open(save_path, 'w') as f:
            json.dump(to_save, f)

    def load_dataset(self, load_path):
        self.storage_path = load_path
        with open(load_path, 'r') as f:
            saved = json.load(f)
            self.full_dataset = saved['full_dataset']
            self.len_dataset = saved['len_dataset']

    def create_eval_problem_set(self, max_num_problems_per_category=100):
        flattened_dataset = {}
        for num_rules in self.full_dataset:
            rule_segment = self.full_dataset[num_rules]

            for rule_instance in rule_segment:
                attributes = rule_instance['attributes']
                attribute_to_rule = rule_instance['attribute_to_rule']
                attribute_to_values = rule_instance['attribute_to_values']
                num_nonconstant_rules = rule_instance['num_nonconstant_rules']

                for problem_prompt, problem_answer, problem_abstraction in zip(rule_instance['problem_prompts'],
                                                                               rule_instance['problem_answers'],
                                                                               rule_instance[
                                                                                   'problem_abstractions']):
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
                    actual_num_rules = len(attribute_to_rule.values())
                    temp_list = flattened_dataset.get(actual_num_rules, [])
                    temp_list.append(eval_problem)
                    flattened_dataset[actual_num_rules] = temp_list

        eval_problems = []
        for num_rules in flattened_dataset:
            print(num_rules, len(flattened_dataset[num_rules]))
            eval_problems += random.sample(flattened_dataset[num_rules], k=min(len(flattened_dataset[num_rules]), max_num_problems_per_category))
            print(len(eval_problems))

        self.eval_dataset = eval_problems
        self.eval_dataset_len = len(eval_problems)
        print('Eval dataset length:', self.eval_dataset_len)

        path_base = self.storage_path
        self.eval_storage_path = path_base[:-5] + '_eval_problems_' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.json'

        with open(self.eval_storage_path, 'w') as f:
            for problem in eval_problems:
                f.write(json.dumps(problem) + '\n')

        gpt4_tokenzier = tiktoken.encoding_for_model('gpt-4')
        eval_metadata = {
            'total_num_rules_count': {},
            'num_nonconstant_rules_count': {},
            'num_distribute_3_rules_count': {},
            'num_unique_rules_count': {},
            'total_input_tokens': {
                'tokenizer': 'tiktoken.encoding_for_model(\'gpt-4\')',
                'num_tokens': sum(len(gpt4_tokenzier.encode(problem['problem_prompt'])) for problem in self.eval_dataset)
            }
        }
        for problem in self.eval_dataset:
            num_rules = len(problem['characteristics']['attribute_to_rule'].values())
            num_nonconstant_rules = len(
                [rule for rule in problem['characteristics']['attribute_to_rule'].values() if
                 rule != 'constant'])
            num_distribute_3_rules = len(
                [rule for rule in problem['characteristics']['attribute_to_rule'].values() if
                 rule == 'distribute_3'])
            num_unique_rules = len(set(problem['characteristics']['attribute_to_rule'].values()))

            eval_metadata['total_num_rules_count'][num_rules] = eval_metadata['total_num_rules_count'].get(num_rules, 0) + 1
            eval_metadata['num_nonconstant_rules_count'][num_nonconstant_rules] = eval_metadata[
                'num_nonconstant_rules_count'].get(
                num_nonconstant_rules, 0) + 1
            eval_metadata['num_distribute_3_rules_count'][num_nonconstant_rules] = eval_metadata[
                'num_distribute_3_rules_count'].get(
                num_distribute_3_rules, 0) + 1
            eval_metadata['num_unique_rules_count'][num_unique_rules] = eval_metadata['num_unique_rules_count'].get(
                num_unique_rules, 0) + 1

        self.eval_metadata = eval_metadata
        with open(self.eval_storage_path[:-5] + '_meta_data.json', 'w') as f:
            json.dump(self.eval_metadata, f)
        ppr.pprint(self.eval_metadata)

    def load_eval_dataset(self, load_path, metadata_path):
        self.eval_storage_path = load_path
        with open(load_path, 'r') as f:
            self.eval_dataset = []
            for item in f:
                self.eval_dataset.append(json.loads(item))

            self.eval_dataset_len = len(self.eval_dataset)

        with open(metadata_path, 'r') as f:
            self.eval_metadata = json.load(f)


def create_text_rpm_dataset(min_num_rules, max_num_rules, max_num_problems_per_category):
    dataset = RPMDataset(min_rules=min_num_rules, max_rules=max_num_rules)
    dataset.generate_dataset()

    new_dataset = RPMDataset()
    new_dataset.load_dataset('default_rpm_dataset.json')
    print('Dataset num problems:', dataset.len_dataset)
    print('Saved properly:', dataset.full_dataset == new_dataset.full_dataset, dataset.storage_path == new_dataset.storage_path)

    new_dataset.create_eval_problem_set(max_num_problems_per_category=max_num_problems_per_category)    # Also prints metadata


def main():
    max_num_problems_per_category = 200
    min_num_rules = 1
    max_num_rules = 10
    create_text_rpm_dataset(min_num_rules=min_num_rules,
                            max_num_rules=max_num_rules,
                            max_num_problems_per_category=max_num_problems_per_category)


if __name__ == '__main__':
    main()
