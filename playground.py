
from rpm import RPMMaker
import dataset
import random
import json

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
        


def main():
    create_example_problem(num_rules=14)


if __name__ == '__main__':
    main()
