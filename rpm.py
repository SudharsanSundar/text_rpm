import random
import pprint as ppr
import itertools

prompt_template1 = '''Consider the following pattern. Each tuple {empty_form_tuple} represents {attribute_tuple}.
Row 1: {row1}
Row 2: {row2}
Row 3: {row3}

Please determine the correct values for the final tuple of Row 3, {mystery_tuple}, which completes the pattern.'''

SUPPORTED_RULES = ['constant', 'progression', 'distribute_3']
SUPPORTED_NUM_UNIQUE_VALUES = 3


# TODO: Tuples might be making things too easy. A more realistic test might be to *concatenate* everything, so you see "ABC" rather than "(A, B, C)". Then, you have to tease it apart, rather than being handed the "parsed" tuple. BUT, this might break tokenization.
class RPMMaker:
    def __init__(self):
        pass

    @staticmethod
    def make_prompt(attributes, problem_abstraction, prompt_template=prompt_template1):
        # Prepare the empty tuple text
        possible_empty_characters = ['_']
        assert len(set(possible_empty_characters) - set(str(problem_abstraction))) > 0
        chosen_empty_character = list(set(possible_empty_characters) - set(str(problem_abstraction)))[0]
        empty_form_values = [chosen_empty_character] * len(attributes)
        mystery_tuple = ['?'] * len(attributes)

        # Prepare the answer text and prompt text
        answer = RPMMaker.format_elem(problem_abstraction[2][2])
        cleaned_attributes = [attribute.replace('_', ' ') for attribute in attributes]
        prompt = prompt_template.format(
            empty_form_tuple=RPMMaker.format_elem(empty_form_values),
            attribute_tuple=RPMMaker.format_elem(cleaned_attributes),
            row1=RPMMaker.format_row(problem_abstraction[0]),
            row2=RPMMaker.format_row(problem_abstraction[1]),
            row3=RPMMaker.format_row(problem_abstraction[2][:-1] + [mystery_tuple]),
            mystery_tuple=RPMMaker.format_elem(mystery_tuple)
        )

        return prompt, answer

    @staticmethod
    def format_row(row_vals):
        return ', '.join(['(' + ', '.join([str(val) for val in elem]) + ')' for elem in row_vals])

    @staticmethod
    def format_elem(elem):
        return '(' + ', '.join([str(val) for val in elem]) + ')'

    @staticmethod
    def generate_random_problem(attribute_to_rule: dict,
                                attribute_to_values: dict = None):
        """
        :param attribute_to_rule: dict, what rule to apply to the given attribute
        :param attribute_to_values: optional dict, what values to use for the given attribute

        :return: 8 train and 1 test matrix, (problem characteristics as well?)

        supported rules: (for now, minimal)
        - constant
        - progression
        - distribute 3

        possible attributes in line with traditional RPM:
        - (outside_)shape_type
        - (outside_)shape_size
        - (outside_)shape_color
        - inside_shape_type
        - inside_shape_size
        - inside_shape_color
        [haven't yet thought about how to incorporate inside_grids, but I think it's just more of the same]
        """
        attributes = list(attribute_to_rule.keys())
        assert len(set(attributes)) == len(attributes)

        # Get the 3x3 outputs for each attribute individually
        assert len(set([rule for rule in attribute_to_rule.values()]) - set(SUPPORTED_RULES)) == 0
        rule_outputs = {}
        for attribute in attributes:
            rule_output = RPMMaker.apply_rule_to_attribute(attribute_to_rule[attribute])
            rule_outputs[attribute] = rule_output

        # Create abstraction for the final matrix
        final_grid_abstraction = RPMMaker.create_final_grid_abstraction(rule_outputs, attributes, attribute_to_rule, attribute_to_values)

        return attributes, final_grid_abstraction

    @staticmethod
    def apply_rule_to_attribute(rule, order=None, sign=None):
        if rule == 'constant':
            row_seeds = random.sample([i for i in range(3)], k=3) if order is None else order
            return [[row_seeds[i] for j in range(3)] for i in range(3)]
        elif rule == 'progression':
            progression = random.sample([i for i in range(3)], k=3) if order is None else order
            return [[progression[j] for j in range(3)] for i in range(3)]
        elif rule == 'distribute_3':
            progression = random.sample([i for i in range(3)], k=3) if order is None else order
            sign = random.choice([-1, 1]) if sign is None else sign
            return [[progression[(j + sign * i) % 3] for j in range(3)] for i in range(3)]
        else:
            raise NotImplementedError(
                f'Haven\'t designed the rule {rule} and implemented logic for it to use in a matrix.')

    @staticmethod
    def create_final_grid_abstraction(rule_outputs, attributes, attribute_to_rule, attribute_to_values):
        # Merge all attributes to create abstraction for the final matrix
        final_grid_abstraction = [[[rule_outputs[attribute][i][j] for attribute in attributes] for j in range(3)] for i
                                  in range(3)]

        # Map indices to custom values/scales for each attribute if desired
        if attribute_to_values is not None:
            assert attribute_to_values.keys() == attribute_to_rule.keys()
            assert all(len(values) == 3 for values in
                       attribute_to_values.values())  # We currently only allow values for a given attribute to take on 3 unique values
            assert all(all("?" not in value for value in values) for values in
                       attribute_to_values.values())  # We reserve "?" as a special character
            final_grid_abstraction = [
                [[attribute_to_values[attributes[k]][final_grid_abstraction[i][j][k]] for k in range(len(attributes))]
                 for j in range(3)] for i in range(3)]

        return final_grid_abstraction

    @staticmethod
    def generate_all_unique_problems_from_rules(attribute_to_rule: dict,
                                                attribute_to_values: dict = None,
                                                max_num_rule_configs_tried: int = 1000):
        """
        :param attribute_to_rule: dict, what rule to apply to the given attribute
        :param attribute_to_values: optional dict, what values to use for the given attribute
        :param max_num_rule_configs_tried: maximum number of alternative configurations (i.e. random seeding of each rule "token") of the specified problem to try

        :return: attribute list, list of RPM abstractions

        supported rules: (for now, minimal)
        - constant
        - progression
        - distribute_3

        possible attributes in line with traditional RPM:
        - (outside_)shape_type
        - (outside_)shape_size
        - (outside_)shape_color
        - inside_shape_type
        - inside_shape_size
        - inside_shape_color
        [haven't yet thought about how to incorporate inside_grids, but I think it's just more of the same]
        """
        attributes = list(attribute_to_rule.keys())
        assert len(set(attributes)) == len(attributes)

        total_num_unique = 6 ** len(attributes) * 2 ** sum(1 for attribute in attributes if attribute_to_rule[attribute] == 'distribute_3')     # Relies on the fact that there are only 3 unique values per attribute in the entire matrix

        all_abstractions = []
        rule_configs = RPMMaker.get_all_rule_configs(attributes, attribute_to_rule, max_num_rule_configs_tried)   # Relies on the fact that there are only 3 unique values per attribute
        assert total_num_unique >= len(rule_configs)        # Allows for capping the max tried. In smaller scale testing, worked fine.

        # Go through each possible unique configuration/application of the rules
        assert len(set([rule for rule in attribute_to_rule.values()]) - set(SUPPORTED_RULES)) == 0
        for rule_config in rule_configs:
            # Get the 3x3 outputs for each attribute individually
            rule_outputs = {}
            for attribute in attributes:
                rule_output = RPMMaker.apply_rule_to_attribute(
                    attribute_to_rule[attribute],
                    order=rule_config[attribute]['order'],
                    sign=None if attribute_to_rule[attribute] != 'distribute_3' else rule_config[attribute]['sign']
                )

                rule_outputs[attribute] = rule_output

            # Create abstraction for the final matrix
            final_grid_abstraction = RPMMaker.create_final_grid_abstraction(rule_outputs, attributes, attribute_to_rule,
                                                                            attribute_to_values)
            all_abstractions.append(final_grid_abstraction)

        # all_abstractions = RPMMaker.get_rid_of_duplicate_items(all_abstractions)

        return attributes, all_abstractions

    @staticmethod
    def get_all_rule_configs(attributes, attribute_to_rule, max_num_configs):
        all_configs = []

        def compute_config_variations(attr_idx, config, configs):
            if len(configs) >= max_num_configs:
                return
            if attr_idx == len(attributes):
                configs.append(config)
            else:
                attribute = attributes[attr_idx]
                order_permutations = list(itertools.permutations([i for i in range(3)]))
                random.shuffle(order_permutations)

                if attribute_to_rule[attributes[attr_idx]] != 'distribute_3':
                    for order_permutation in order_permutations:
                        new_config = {} if config is None else config.copy()
                        new_config[attribute] = {}
                        new_config[attribute]['order'] = order_permutation

                        compute_config_variations(attr_idx + 1, new_config, configs)
                elif attribute_to_rule[attributes[attr_idx]] == 'distribute_3':
                    sign_permutations = [-1, 1]
                    for order_permutation in order_permutations:
                        for sign_permutation in sign_permutations:
                            new_config = {} if config is None else config.copy()
                            new_config[attribute] = {}
                            new_config[attribute]['order'] = order_permutation
                            new_config[attribute]['sign'] = sign_permutation

                            compute_config_variations(attr_idx + 1, new_config, configs)

        compute_config_variations(0, None, all_configs)

        return all_configs

    @staticmethod
    def get_rid_of_duplicate_items(items):
        final_items = []
        for item in items:
            if item not in final_items:
                final_items.append(item)

        return final_items


def check_all_items_unique(items):
    num_duplicates = 0
    for itemA in items:
        num_same = 0
        for itemB in items:
            num_same += 1 if itemA == itemB else 0

        if num_same != 1:
            print(f'Found duplicate item:\n{itemA}, num occurences: {num_same}')
            num_duplicates += 1

    print('-'*100)
    print(f'Finished checking for duplicates. Total num items: {len(items)}. Total num duplicates found: {num_duplicates}')
    print('-' * 100)


def main():
    maker = RPMMaker()
    attributes, problem_abstraction = maker.generate_random_problem(
        attribute_to_rule={'shape_type': 'progression',
                           'shape_color': 'constant',
                           'shape_size': 'progression'},
        attribute_to_values={'shape_type': ['!', 'A', '#'],
                             'shape_color': ['D', '@', 'F'],
                             'shape_size': ['&', 'H', '^']},
    )

    prompt, answer = maker.make_prompt(attributes, problem_abstraction)
    print(prompt)
    print(answer)

    attribute_seq, all_unique_problems = maker.generate_all_unique_problems_from_rules(
        attribute_to_rule={'shape_type': 'progression',
                           'shape_color': 'constant',
                           'shape_size': 'progression'},
    )

    check_all_items_unique(all_unique_problems)


if __name__ == '__main__':
    main()
