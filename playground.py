
from rpm import RPMMaker
import dataset
import random
import json
import numpy as np
from typing import List
from pprint import pprint as ppr
import math

DEFAULT_ALPHABET = tuple([[chr(65 + i + 3 * j) for i in range(3)] for j in range(8)] + 
                         [[chr(97 + i + 3 * j) for i in range(3)] for j in range(8)]) 

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


class PatternSeq:
    def __init__(self,
                 base_seqs: List[List[int]], 
                 depth: int, 
                 num_repeats_before_pred: int,
                 num_pred: int=3) -> None:
        '''
        num base seqs [1, ...]
        base seq len(s) [1, ...]
        base seq inter-div (prefix-free? also not sure how to measure this)
        base seq vocab (intra-div) [[0,1], [-1,0], [-1,1] [-1,0,1], [0,1,2], ...]
        depth [0, ...]

        depth operator diversity [cumsum, cumprod, subset sum, subset prod, flip, negative, etc.?]
        
        '''
        self.num_base_seqs = base_seqs
        self.depth = depth
        self.num_repeats_before_pred = num_repeats_before_pred
        self.num_pred = num_pred

    @staticmethod
    def num_seq_stack(base_seqs: List[List[int]], 
                      depth: int, 
                      num_repeats_before_pred: int,
                      num_pred: int=3):
        assert depth >= 0

        longest_base_seq_len = max(len(base_seq) for base_seq in base_seqs)
        num_extra_digits = num_pred + depth
        num_extra_repeats = round_up(num_extra_digits / longest_base_seq_len)
        num_to_cut = num_extra_digits % longest_base_seq_len
        final_seq_len = longest_base_seq_len * (num_repeats_before_pred + num_extra_repeats) - num_to_cut
        # NOTE: One difficulty metric: the number of unique vertical sequences formed by stack of base seqs = how mutually diverse they are

        final_seq = np.zeros(final_seq_len, dtype=int)

        for base_seq in base_seqs:
            temp_seq = np.array(base_seq * round_up(final_seq_len / len(base_seq)), dtype=int)
            temp_seq = temp_seq[:final_seq_len]

            for i in range(depth):
                temp_seq = np.cumsum(temp_seq, dtype=int)
            
            final_seq += temp_seq
            
        return list(final_seq[:-num_pred]), list(final_seq[-num_pred:])


def generate_all_seq_combos(max_len=4, moveset=[0, 1]):
    def generate_combos(combo_len, combo):
        if len(combo) == combo_len:
            combos[combo_len].append(combo)
        else:
            for move in moveset:
                generate_combos(combo_len, combo + [move])
    
    all_combos = []
    combos = {i: [] for i in range(1, max_len + 1, 1)}
    for i in range(1, max_len + 1, 1):
        generate_combos(i, [])
    
    def generate_combo_combos(ccombo):
        if len(ccombo) == max_len:
            all_combos.append(ccombo)
        else:
            for combo in combos[len(ccombo) + 1]:
                generate_combo_combos(ccombo + [combo])
    
    generate_combo_combos([])
    return all_combos


def main():
    all_4_seq_combos = generate_all_seq_combos(max_len=4)

    for i in range(10):
        sampled_base_seqs = random.choice(all_4_seq_combos)
        seq, ans = PatternSeq.num_seq_stack(
            base_seqs=sampled_base_seqs,
            depth=0,
            num_repeats_before_pred=3
        )
        print(str(seq)[1:-1], '->', str(ans)[1:-1])
        print(f'({sampled_base_seqs})')
        print('-----')


if __name__ == '__main__':
    main()
