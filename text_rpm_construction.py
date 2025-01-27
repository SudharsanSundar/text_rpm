import numpy as np
from typing import Any, List, Dict
from pprint import pprint as ppr
import random

SUPPORTED_RULE_TYPES = ['dependent', 'independent']


class RPMElement:
    def __init__(self,
                 attr_name: str,
                 attr_domain: List[str],
                 loc: List[int]) -> None:
        assert len(attr_domain) > 0
        assert len(loc) == 2 and type(loc[0]) == type(loc[1]) and type(loc[1]) == int
        
        self.name = attr_name
        self.domain = attr_domain
        self.loc = loc
        self.value = None


class RPMRule:
    '''
    Wrapper class for rules/patterns used to vary element values across a rows of an RPM problem.

    Sets structure of basic rule parameters and rule application return type.
    '''
    def __init__(self,
                 attr_names: List[str],
                 rule_name: str,
                 num_rows: int,
                 num_cols: int) -> None:
        self.attr_names = attr_names
        self.rule_name = rule_name
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.attr_to_values = {attr_name: [[None for j in range(self.num_cols)] for i in range(self.num_rows)] for attr_name in self.attr_names}
    
    def apply_rule() -> Dict[str, List[List[int]]]:
        pass


class ConstantRowRule(RPMRule):
    '''
    Constant in a row rule, e.g.:
    000
    111
    222

    Valid ordering is, e.g.: {'order': range(self.num_cols)}
    '''
    def __init__(self, 
                 attr_names: List[str],
                 rule_name: str,
                 num_rows: int,
                 num_cols: int) -> None:
        assert len(attr_names) == 1
        super().__init__(attr_names, rule_name, num_rows, num_cols)
    
    def apply_rule(self, ordering=None) -> Dict[str, List[List[int]]]:
        if ordering is None:
            ordering = {'order': range(self.num_cols)}
        else:
            assert 'order' in ordering \
                and len(ordering.keys()) == 1 \
                and type(ordering['order']) == list \
                and all(type(elem) == int for elem in ordering['order']) \
                and len(ordering['order']) == self.num_cols

        self.attr_to_values[self.attr_names[0]] = [[ordering['order'][i] for j in range(self.num_cols)] for i in range(self.num_rows)]
        return self.attr_to_values


class ConstantColRule(RPMRule):
    '''
    Constant in a column rule, e.g.:
    012
    012
    012

    Valid ordering is, e.g.: {'order': range(self.num_cols)}
    '''
    def __init__(self, 
                 attr_names: List[str],
                 rule_name: str,
                 num_rows: int,
                 num_cols: int) -> None:
        assert len(attr_names) == 1
        super().__init__(attr_names, rule_name, num_rows, num_cols)
    
    def apply_rule(self, ordering=None) -> Dict[str, List[List[int]]]:
        if ordering is None:
            ordering = {'order': range(self.num_cols)}
        else:
            assert 'order' in ordering \
                and len(ordering.keys()) == 1 \
                and type(ordering['order']) == list \
                and all(type(elem) == int for elem in ordering['order']) \
                and len(ordering['order']) == self.num_cols

        self.attr_to_values[self.attr_names[0]] = [[ordering['order'][j] for j in range(self.num_cols)] for i in range(self.num_rows)]
        return self.attr_to_values
    

class CycleN(RPMRule):
    '''
    Cycle through n values, starting at a different point in the cycle for each row, e.g.:
    012
    120
    201

    Valid ordering is, e.g.: {'order': range(self.num_cols), 'sign': 1 OR -1}
    '''
    def __init__(self, 
                 attr_names: List[str],
                 rule_name: str,
                 num_rows: int,
                 num_cols: int) -> None:
        assert len(attr_names) == 1
        super().__init__(attr_names, rule_name, num_rows, num_cols)
    
    def apply_rule(self, ordering=None) -> Dict[str, List[List[int]]]:
        if ordering is None:
            ordering = {'order': range(self.num_cols), 'sign': 1}
        else:
            assert 'order' in ordering \
                and type(ordering['order']) == list \
                and all(type(elem) == int for elem in ordering['order']) \
                and len(ordering['order']) == self.num_cols \
                and 'sign' in ordering \
                and type(ordering['sign']) == int \
                and (ordering['sign'] == 1 or ordering['sign'] == -1) \
                and len(ordering.keys()) == 2

        self.attr_to_values[self.attr_names[0]] = [[ordering['order'][(i + ordering['sign'] * j) % self.num_cols] for j in range(self.num_cols)] for i in range(self.num_rows)]
        return self.attr_to_values


class CycleNminus1(RPMRule):
    '''
    Cycle through n-1 values, starting at a different point in the cycle for each row, e.g.:
    010
    100
    001

    Valid ordering is, e.g.: {'order': range(self.num_cols), 'sign': 1 OR -1}
    TODO: Make sure this is valid by trying a few problems.
    '''
    def __init__(self, 
                 attr_names: List[str],
                 rule_name: str,
                 num_rows: int,
                 num_cols: int) -> None:
        assert len(attr_names) == 1
        super().__init__(attr_names, rule_name, num_rows, num_cols)
    
    def apply_rule(self, ordering=None) -> Dict[str, List[List[int]]]:
        if ordering is None:
            ordering = {'order': range(self.num_cols), 'sign': -1, 'shift': 1}
        else:
            assert 'order' in ordering \
                and type(ordering['order']) == list \
                and all(type(elem) == int for elem in ordering['order']) \
                and len(ordering['order']) == self.num_cols \
                and 'sign' in ordering \
                and type(ordering['sign']) == int \
                and (ordering['sign'] == 1 or ordering['sign'] == -1) \
                and 'shift' in ordering \
                and type(ordering['shift']) == int \
                and 0 <= ordering['shift'] <= 2 \
                and len(ordering.keys()) == 3

        chosen_cells = []
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if (ordering['sign'] * i + j - ordering['shift']) % self.num_cols == 0:
                    chosen_cells.append([i, j])

        self.attr_to_values[self.attr_names[0]] = [[ordering['order'][int([i, j] in chosen_cells)] for j in range(self.num_cols)] for i in range(self.num_rows)]
        return self.attr_to_values


class Diagonals(RPMRule):
    '''
    Diagonals are one color depending on how far they are from the center diagonal, e.g.:
    012
    101
    210

    (Alternatively, this describes the double-ended sequence 21012)

    Valid ordering is, e.g.: {'order': range(self.num_cols), 'sign': -1 OR 1}
    '''
    def __init__(self, 
                 attr_names: List[str],
                 rule_name: str,
                 num_rows: int,
                 num_cols: int) -> None:
        assert len(attr_names) == 1
        assert num_rows == num_cols
        super().__init__(attr_names, rule_name, num_rows, num_cols)
    
    def apply_rule(self, ordering=None) -> Dict[str, List[List[int]]]:
        if ordering is None:
            ordering = {'order': range(self.num_cols), 'sign': 1}
        else:
            assert 'order' in ordering \
                and type(ordering['order']) == list \
                and all(type(elem) == int for elem in ordering['order']) \
                and len(ordering['order']) == self.num_cols \
                and 'sign' in ordering \
                and type(ordering['sign']) == int \
                and (ordering['sign'] == -1 or ordering['sign'] == 1) \
                and len(ordering.keys()) == 2

        self.attr_to_values[self.attr_names[0]] = [[abs(i + ordering['sign'] * j - int(ordering['sign'] == 1) * (self.num_cols - 1)) for j in range(self.num_cols)] for i in range(self.num_rows)]
        if ordering['sign'] == -1:
            # self.attr_to_values[self.attr_names[0]] = [[abs(i - j) for j in range(self.num_cols)] for i in range(self.num_rows)]
            self.attr_to_values[self.attr_names[0]] = [[ordering['order'][self.attr_to_values[self.attr_names[0]][i][j]] for j in range(self.num_cols)] for i in range(self.num_rows)]
        elif ordering['sign'] == 1:
            # self.attr_to_values[self.attr_names[0]] = [[abs(i + j - self.num_cols + 1) for j in range(self.num_cols)] for i in range(self.num_rows)]
            self.attr_to_values[self.attr_names[0]] = [[ordering['order'][self.attr_to_values[self.attr_names[0]][i][j]] for j in range(self.num_cols)] for i in range(self.num_rows)]
        
        return self.attr_to_values


class GeneralCycle(RPMRule):
    def __init__(self, 
                 attr_names: List[str] | None, 
                 rule_name: str, 
                 num_rows: int, 
                 num_cols: int,
                 path: List[int]) -> None:
        if attr_names is None:
            attr_names = [None] * num_cols
        super().__init__(attr_names, rule_name, num_rows, num_cols)

        self.path = path
    
    def __call__(self, attr_names, rule_name, num_rows, num_cols) -> Any:
        return GeneralCycle(attr_names, rule_name, num_rows, num_cols, path=self.path)

    def apply_rule(self, ordering) -> Dict[str, List[List[int]]]:
        if ordering is None:
            ordering = {'order': range(self.num_cols), 'row_starts': range(self.num_rows)}
        else:
            assert 'order' in ordering \
                and type(ordering['order']) == list \
                and all(type(elem) == int for elem in ordering['order']) \
                and len(ordering['order']) == self.num_cols \
                and 'row_starts' in ordering \
                and type(ordering['row_starts']) == list \
                and all(type(elem) == int for elem in ordering['row_starts']) \
                and len(ordering.keys()) == 2
        
        row_starts = ordering['row_starts'] 
        grid = []
        for row_start in row_starts:
            grid.append([ordering['order'][(row_start + path_pos) % self.num_cols] for path_pos in self.path])
        
        self.attr_to_values[self.attr_names[0]] = grid
        return self.attr_to_values


class GeneralCycle2(RPMRule):
    def __init__(self, 
                 attr_names: List[str] | None, 
                 rule_name: str, 
                 num_rows: int, 
                 num_cols: int,
                 rel_path: List[int]) -> None:
        if attr_names is None:
            attr_names = [None] * num_cols
        super().__init__(attr_names, rule_name, num_rows, num_cols)

        self.rel_path = rel_path
    
    def __call__(self, attr_names, rule_name, num_rows, num_cols) -> Any:
        return GeneralCycle2(attr_names, rule_name, num_rows, num_cols, rel_path=self.rel_path)

    def apply_rule(self, ordering) -> Dict[str, List[List[int]]]:
        if ordering is None:
            ordering = {'order': range(self.num_cols), 'row_starts': range(self.num_rows)}
        else:
            assert 'order' in ordering \
                and type(ordering['order']) == list \
                and all(type(elem) == int for elem in ordering['order']) \
                and len(ordering['order']) == self.num_cols \
                and 'row_starts' in ordering \
                and type(ordering['row_starts']) == list \
                and all(type(elem) == int for elem in ordering['row_starts']) \
                and len(ordering.keys()) == 2
        
        row_starts = ordering['row_starts']
        paths = []
        for i in range(len(self.rel_path) + 1): # This is a fixed pattern. Could vary this too, but don't have time to implement rn
            temp_path = self.rel_path[i:] + self.rel_path[:i]
            paths.append([int(elem) for elem in np.cumsum(temp_path, dtype=int)])

        grid = []
        for row_start, path in zip(row_starts, paths):
            grid.append([ordering['order'][(row_start + path_pos) % self.num_cols] for path_pos in path])
                
        self.attr_to_values[self.attr_names[0]] = grid
        return self.attr_to_values


def generate_all_cycle_rules(attr_names=None, n=5, l=5, balance_with_binary=True, balance_factor=1.25):
    def generate_paths(lim_n, it, path):
        if it == lim_n:
            all_paths.append(path)
        else:
            for elem in [-1, 0, 1]:
                generate_paths(lim_n, it + 1, path + [elem])

    all_paths = []
    generate_paths(l - 1, 0, [0])
    absolute_paths = [list(np.cumsum(item)) for item in all_paths]

    general_cycle_rules = {}
    for i, path in enumerate(absolute_paths):
        general_cycle_rules[f'general_cycle_{i}'] = GeneralCycle(
            attr_names=attr_names, 
            rule_name=f'general_cycle_{i}', 
            num_rows=n, 
            num_cols=l,
            path=path
        )
    
    if balance_with_binary:
        num_nary_rules = len(generate_nary_rules(attr_names=None, n=n, l=n, a_len=l, b_len=l)) 
        num_orig_rules = len(general_cycle_rules)
        kept_cycles = random.sample(list(general_cycle_rules.keys()), k=min(len(general_cycle_rules.keys()), int(num_nary_rules * balance_factor)))
        general_cycle_rules = {key: general_cycle_rules[key] for key in kept_cycles}
    
        print(f'generated {len(general_cycle_rules)} general cycle rules, given {num_nary_rules} nary rules ({num_nary_rules * balance_factor} factor multiplied) and {num_orig_rules} base rules')
    
    return general_cycle_rules


def generate_all_cycle2_rules(attr_names=None, n=5, l=5, balance_with_binary=True, balance_factor=1.25):
    def generate_paths(lim_n, it, path):
        if it == lim_n:
            all_paths.append(path)
        else:
            for elem in [-1, 0, 1]:
                generate_paths(lim_n, it + 1, path + [elem])

    all_paths = []
    generate_paths(l - 1, 0, [0])

    general_cycle_rules = {}
    for i, path in enumerate(all_paths):
        general_cycle_rules[f'general_cycle2_{i}'] = GeneralCycle2(
            attr_names=attr_names, 
            rule_name=f'general_cycle2_{i}', 
            num_rows=n, 
            num_cols=l,
            rel_path=path
        )
        print(i, path)
    
    if balance_with_binary:
        num_nary_rules = len(generate_nary_rules(attr_names=None, n=n, l=n, a_len=l, b_len=l)) 
        num_orig_rules = len(general_cycle_rules)
        kept_cycles = random.sample(list(general_cycle_rules.keys()), k=min(len(general_cycle_rules.keys()), int(num_nary_rules * balance_factor)))
        general_cycle_rules = {key: general_cycle_rules[key] for key in kept_cycles}

        print(f'generated {len(general_cycle_rules)} general cycle rules, given {num_nary_rules} nary rules ({num_nary_rules * balance_factor} factor multiplied) and {num_orig_rules} base rules')
    
    return general_cycle_rules


class MappingOperation:
    def __init__(self, num_rows: int, num_cols: int, operation: Any=lambda x, y: int(x == y), func_type: str='lambda') -> None:
        self.func_type = func_type
        self.operation = operation

    def __call__(self, args: List[int]) -> Any:
        if self.func_type == 'lambda':
            return self.operation(*args)
        elif self.func_type == 'lookup':
            return self.operation(tuple(args))


class NaryRule(RPMRule):
    def __init__(self, 
                 attr_names: List[str] | None, 
                 rule_name: str, 
                 num_rows: int, 
                 num_cols: int,
                 mapping: MappingOperation) -> None:
        if attr_names is None:
            attr_names = [None] * num_cols
        super().__init__(attr_names, rule_name, num_rows, num_cols)
        self.mapping = mapping
    
    def __call__(self, 
                 attr_names: List[str], 
                 rule_name: str, 
                 num_rows: int, 
                 num_cols: int) -> Any:
        return NaryRule(attr_names, rule_name, num_rows, num_cols, mapping=self.mapping)
    
    def apply_rule(self, input_grids, ordering=None) -> Dict[str, List[List[int]]]:
        assert ordering is None
        assert len(input_grids) > 1

        new_grid = []
        for i in range(self.num_rows):
            new_row = []
            for j in range(self.num_cols):
                new_row.append(self.mapping([int(grid[i][j]) for grid in input_grids]))
            
            new_grid.append(new_row)
        
        self.attr_to_values[self.attr_names[-1]] = new_grid

        return self.attr_to_values


def generate_nary_rules(attr_names=None, n=5, l=5, a_len=5, nary=2, b_len=5):
    # # Simple: Only basic logic- or arithmetic-based mappings to full, ternary, or binary alphabet
    assert nary == 2
    assert b_len == a_len
    assert a_len >= 3

    mid = int(a_len / 2)
    operations = [
        lambda x, y: min(x + y, a_len - 1), # up to ceil ; 0 to a_len-1 ; archetype: arithmetic tree, w min max etc., all mod a_len
        lambda x, y: abs(x - y), # abs difference
        lambda x, y: int((x + y) / 2), # avg
        lambda x, y: int(abs(x - y) / 2), # abs difference from avg
        lambda x, y: max(x, y), # max
        lambda x, y: int(max(x, y) / 2), # half the max
        lambda x, y: min(x, y), # min
        lambda x, y: int(min(x, y) * 2) % a_len, # twice the min mod a_len
        lambda x, y: (x + y) % a_len, # addition on field
        lambda x, y: (x * y) % a_len, # multiplication on field
        lambda x, y: int(x >= mid) + int(y >= mid), # 012
        lambda x, y: int(x >= mid) + int(y <= mid),
        lambda x, y: int(x <= mid) + int(y <= mid),
        lambda x, y: int(x <= mid) + int(y >= mid),
        lambda x, y: int(x == y), # 01
        lambda x, y: int(x != y),
        lambda x, y: x >= y,
        lambda x, y: x <= y,
        lambda x, y: x >= mid and y >= mid,
        lambda x, y: x <= mid and y <= mid,
        lambda x, y: (x >= mid and y <= mid) or (x <= mid and y >= mid),
        lambda x, y: x >= mid or y >= mid,
        lambda x, y: x <= mid or y <= mid,
    ]

    nary_structs = {f'nary_{i}': NaryRule(None, f'nary_{i}', n, l, MappingOperation(n, l, operation)) for i, operation in enumerate(operations)}

    return nary_structs


DEFAULT_RULE_CONSTRUCTS = {
    'constant_row': ConstantRowRule,
    'constant_col': ConstantColRule,
    'cycle_n': CycleN,
    'cycle_n_minus_1': CycleNminus1,
    'diagonals': Diagonals,
    'general_cycle': GeneralCycle
}


class RPMProblem:
    '''
    Class representing a single Text RPM problem.
    '''
    def __init__(self, 
                 num_rows: int, 
                 num_cols: int, 
                 attr_names: List[str],
                 attr_domains: Dict[str, List[str]],
                 rule_to_attr: List[str | List[str]],
                 rule_maps: List[str | List[int | None]],
                 rule_to_ordering: List[List[str | Dict[str, List[int]]]]=None,
                 rule_constructs: Dict[str, RPMRule]=DEFAULT_RULE_CONSTRUCTS,
                 ) -> None:
        # Make sure problem parameters are valid
        assert num_rows > 1
        assert num_cols > 1
        assert len(attr_names) > 0

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_attrs_per_cell = len(attr_names)
        self.attr_names = attr_names
        self.attr_domains = attr_domains
        self.rule_to_attr = rule_to_attr
        self.rule_to_ordering = rule_to_ordering
        self.rule_maps = rule_maps

        # Create RPM grid which holds a literal instance of the problem
        self.grid = [
            [
                [
                    None for k in range(self.num_attrs_per_cell)
                ] for j in range(self.num_cols)
            ] for i in range(self.num_rows)
        ]

        # Check to make sure rule assignments are valid, then fetch all rule classes used for this problem
        attrs_from_rules = []
        for pair in self.rule_to_attr:
            attrs_from_rules.append(*pair[1])
        assert all(pair[0] in rule_constructs for pair in self.rule_to_attr)
        assert all(attr in self.attr_names for attr in attrs_from_rules) and all(attr in attrs_from_rules for attr in self.attr_names)
        assert len(set(attrs_from_rules)) == len(attrs_from_rules)

        self.rules = [rule_constructs[pair[0]](pair[1], pair[0], self.num_rows, self.num_cols) for pair in self.rule_to_attr]
        
        # Calculate and organize value assignments to each attribute based on rules, including remapping to different indices if desired
        assert all(
            pair1[0] == pair2[0] for pair1, pair2 in zip(rule_maps, self.rule_to_attr)
            ) and all(
            type(pair[1]) == list and (all(type(elem) == int for elem in pair[1]) or len(pair[1]) == 0) for pair in rule_maps
        )
        assert self.rule_to_ordering is None or (
            all(pair[0] in [pair2[0] for pair2 in self.rule_to_attr] for pair in self.rule_to_ordering) and 
            all(pair2[0] in [pair[0] for pair in self.rule_to_ordering] for pair2 in self.rule_to_attr) and 
            all(pair[0] == pair2[0] for pair, pair2 in zip(self.rule_to_attr, rule_to_ordering))
        )
        if not all(len(rule_maps[i][1]) == 0 for i, rule in enumerate(self.rules) if rule.rule_name[:len('nary')] != 'nary'):
            print(rule_maps)
            print([len(rule_maps[i][1]) == 0 for i, rule in enumerate(self.rules) if rule.rule_name[:len('nary')] != 'nary'])
            assert all(len(rule_maps[i][1]) == 0 for i, rule in enumerate(self.rules) if rule.rule_name[:len('nary') != 'nary'])
        assert all(len(rule_maps[i][1]) > 1 for i, rule in enumerate(self.rules) if rule.rule_name[:len('nary')] == 'nary')
        
        # !! Assumes each rule yields values for exactly 1 attribute !!

        rules_to_values = [None] * len(self.rules)
        it = 0
        while None in rules_to_values:  # Number of iterations should equal maximum dependency depth
            it += 1
            for i, rule in enumerate(self.rules):
                if len(rule_maps[i][1]) == 0:   # Independent rules
                    rule_output = rule.apply_rule(ordering=self.rule_to_ordering[i][1] if self.rule_to_ordering is not None else self.rule_to_ordering)
                    rules_to_values[i] = rule_output
                elif len(rule_maps[i][1]) > 1:  # Dependent rules
                    input_grids = []
                    for idx in rule_maps[i][1]:
                        if rules_to_values[idx] is not None:
                            for attr in rules_to_values[idx]:
                                input_grids.append(rules_to_values[idx][attr])
                        else:
                            input_grids.append(None)
                    
                    if None not in input_grids: # If last layer of dependencies has been filled, fill in the given dependent rule
                        rule_output = rule.apply_rule(ordering=self.rule_to_ordering[i][1] if self.rule_to_ordering is not None else self.rule_to_ordering,
                                                      input_grids=input_grids)
                        rules_to_values[i] = rule_output
            
            if it > 20: # TODO: Debugging
                print(rules_to_values)
                print(rule_maps)

                assert False

        
        attr_to_values = {}
        for i, rule in enumerate(self.rules):   # Sync rule outputs to attr_to_values
            for attr in rules_to_values[i]:
                attr_to_values[attr] = rules_to_values[i][attr]

        # Assign RPM grid elements their values based on the given rules
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                for k, attr_name in zip(range(self.num_attrs_per_cell), self.attr_names):
                    self.grid[i][j][k] = self.attr_domains[attr_name][attr_to_values[attr_name][i][j]]
    
    def get_grid(self):
        return self.grid


def main():
    # print('~'*100)
    # test_problem = RPMProblem(
    #     num_rows=5,
    #     num_cols=5,
    #     attr_names=['shape', 'color', 'texture'],
    #     attr_domains={'shape': ['A', 'B', 'C', 'D', 'E'], 'color': ['A', 'B', 'C', 'D', 'E'], 'texture': ['A', 'B', 'C', 'D', 'E']},
    #     rule_to_attr=[['constant_row', ['shape']], ['constant_col', ['color']], ['diagonals', ['texture']]],
    #     rule_to_ordering=[['constant_row', {'order': [0, 1, 2, 3, 4]}], ['constant_col', {'order': [0, 1, 2, 3, 4]}], ['diagonals', {'order': [0, 1, 2, 3, 4], 'sign': 1}]]
    # )
    # print('-'*100)
    # for row in test_problem.grid:
    #     print(row)
    
    # print(len(generate_nary_rules(None, n=5, l=5, a_len=5, nary=2, b_len=5)))
    print(len(generate_all_cycle2_rules(None, n=3, l=3)))


if __name__ == '__main__':
    main()
