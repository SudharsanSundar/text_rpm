import numpy as np
from typing import List, Dict
import pprint as ppr


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
                 mapping_type: str='random',
                 rule_to_ordering: List[List[str | Dict[str, List[int]]]]=None) -> None:
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

        # Create RPM grids, one which holds an abstracted version of the problem, and one which holds a literal instance of the problem
        self.grid_struct = [
            [
                [
                    RPMElement(attr_name=attr_name, attr_domain=self.attr_domains[attr_name], loc=[i, j]) for attr_name in self.attr_names
                ] for j in range(self.num_cols)
            ] for i in range(self.num_rows)
        ]
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
        assert all(pair[0] in RULE_CONSTRUCTS for pair in self.rule_to_attr)
        assert all(attr in self.attr_names for attr in attrs_from_rules) and all(attr in attrs_from_rules for attr in self.attr_names)
        assert len(set(attrs_from_rules)) == len(attrs_from_rules)

        self.rules = [RULE_CONSTRUCTS[pair[0]](pair[1], pair[0], self.num_rows, self.num_cols) for pair in self.rule_to_attr]
        
        # Calculate and organize value assignments to each attribute based on rules, including remapping to different indices if desired
        assert self.rule_to_ordering is None or (
            all(pair[0] in [pair2[0] for pair2 in self.rule_to_attr] for pair in self.rule_to_ordering) and 
            all(pair2[0] in [pair[0] for pair in self.rule_to_ordering] for pair2 in self.rule_to_attr) and 
            all(pair[0] == pair2[0] for pair, pair2 in zip(self.rule_to_attr, rule_to_ordering))
        )
        attr_to_values = {}
        for i, rule in enumerate(self.rules):
            rule_output = rule.apply_rule(ordering=self.rule_to_ordering[i][1] if self.rule_to_ordering is not None else self.rule_to_ordering)

            for attr in rule_output:
                attr_to_values[attr] = rule_output[attr]
        
        # Assign RPM grid elements their values based on the given rules
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                for k, attr_name in zip(range(self.num_attrs_per_cell), self.attr_names):
                    self.grid_struct[i][j][k].value = attr_to_values[attr_name][i][j]
                    self.grid[i][j][k] = self.attr_domains[attr_name][attr_to_values[attr_name][i][j]]
    
    def get_grid(self):
        return self.grid

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
        print(self.attr_names)
    
    def apply_rule() -> Dict[str, List[List[int]]]:
        pass


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


RULE_CONSTRUCTS = {
    'constant_row': ConstantRowRule,
    'constant_col': ConstantColRule,
    'cycle_n': CycleN,
    'cycle_n_minus_1': CycleNminus1,
    'diagonals': Diagonals
}


'''
# : Choose 3? // --> can be degenerate (ok ish?), and ordering is tough (num_cols! ^ num_rows) --> actually I don't like these rules for a variety of reasons

# : Choose 2? // --> can also be degenerate --> also skip. Harder to implement, but also harder to be correct, and might be cool to try double attr rules instead

# : Distractions? --> seems tough to do without screwing up determination  //

# : Answers? Generation is harder, so a better task I think. But can't do distractions without answers. Keep this on backburner for now //

# TODO: [some double attribute rule]

must be intertwined somehow, otherwise you can just decompose it into each individual attr

we can try some arithmetic type stuff, where things are encoded in binary!
but quite complicated, and might be hard to do with 3x3?

--> problem is it's hard to get enough information across in a 3x3 grid

00 00 00
01 00 01
00 01 01
01 01 10
11 01 00
01 11 00
11 11 01

Noise!
01 00 11
00 11 11
00 01 

'''


def main():
    print('~'*100)
    test_problem = RPMProblem(
        num_rows=5,
        num_cols=5,
        attr_names=['shape', 'color', 'texture'],
        attr_domains={'shape': ['A', 'B', 'C', 'D', 'E'], 'color': ['A', 'B', 'C', 'D', 'E'], 'texture': ['A', 'B', 'C', 'D', 'E']},
        rule_to_attr=[['constant_row', ['shape']], ['constant_col', ['color']], ['diagonals', ['texture']]],
        rule_to_ordering=[['constant_row', {'order': [0, 1, 2, 3, 4]}], ['constant_col', {'order': [0, 1, 2, 3, 4]}], ['diagonals', {'order': [0, 1, 2, 3, 4], 'sign': 1}]]
    )
    print('-'*100)
    for row in test_problem.grid:
        print(row)


if __name__ == '__main__':
    main()
