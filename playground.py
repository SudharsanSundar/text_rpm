
from rpm import RPMMaker
import dataset
import random
import json
import numpy as np
from typing import List
from pprint import pprint as ppr
import math

DEFAULT_ALPHABET = tuple([[chr(65 + i + 3 * j) for i in range(3)] for j in range(8)] + 
                         [[chr(97 + i + 3 * j) for i in range(3)] for j in range(8)])   # lowercase letters augmented
# DEFAULT_ALPHABET = tuple([[chr(65 + i + 3 * j) for i in range(3)] for j in range(8)] + 
#                          [[i + 3 * j for i in range(1, 4, 1)] for j in range(6)])   # numbers augmented
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
                      'inner_shape_count')      # at 14 rn
SUPPORTED_RULES = ('constant', 'progression', 'distribute_3')


def create_example_problem(num_rules, num_sampled=3):
    attr_to_values = {attr: val for attr, val in zip(DEFAULT_ATTRIBUTES[:num_rules], DEFAULT_ALPHABET[:num_rules])}
    attr_to_rule = {attr: random.choice(SUPPORTED_RULES) for attr in DEFAULT_ATTRIBUTES[:num_rules]}

    rpmmaker = RPMMaker()

    attributes, problems = rpmmaker.generate_all_unique_problems_from_rules(attr_to_rule, attr_to_values)

    used_abstractions = random.sample(problems, k=num_sampled)
    prompts = [rpmmaker.make_prompt(attributes, abstraction)[0] for abstraction in used_abstractions]
    answers = [rpmmaker.make_prompt(attributes, abstraction)[1] for abstraction in used_abstractions]
    for prompt, answer in zip(prompts, answers):
        print(prompt)
        print('ANSWER', answer)
        print('-'*100)


def check_num_rules_problems(num_rules):
    with open('datasets/default_rpm_dataset_eval_problems_v1.json', 'r') as f:
        eval_problems = []
        for line in f:
            problem = json.loads(line)
            if len(problem['attributes']) == 1:
                eval_problems.append(problem)
        

def round_up(x):
    return int(math.ceil(x))


def num_seq_stack(base_seqs: List[List[int]], 
                  depth: int, 
                  num_repeats_before_pred: int,
                  num_pred: int=3):
    assert depth >= 0

    longest_base_seq_len = max(len(base_seq) for base_seq in base_seqs)
    num_extra_digits = num_pred + depth
    num_extra_repeats = round_up(num_extra_digits / longest_base_seq_len)
    num_to_cut = num_extra_digits % len(base_seq)
    final_seq_len = longest_base_seq_len * (num_repeats_before_pred + num_extra_repeats) - num_to_cut

    final_seq = np.zeros(final_seq_len)

    for base_seq in base_seqs:
        temp_seq = np.array(base_seq * round_up(final_seq_len / len(base_seq)))
        temp_seq = temp_seq[:final_seq_len]

        for i in range(depth):
            temp_seq = np.cumsum(final_seq)
        
        final_seq += temp_seq
        
    return list(final_seq[:-num_pred]), list(final_seq[-num_pred:])


def main():
    for i in range(4):
        seq, ans = num_seq_stack(
            base_seqs=[[1]],
            depth=i,
            num_repeats_before_pred=3
        )
        print(str(seq)[1:-1], '->', str(ans)[1:-1])
        print('-----')


if __name__ == '__main__':
    main()
