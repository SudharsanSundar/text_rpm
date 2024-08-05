from typing import List
import pprint as ppr
import sys
import io
from itertools import chain
import numpy as np
import random

'''
generative options

RPM
1. cycle len == num cols
2. cycle len == num rows + 2 or + 1

Number sequences
1. Path x Path x Path
Have a pattern occuring over certain indices that are layered. 
E.g. one pattern happening on all indices, one happening on even indices, one happening on every 3rd index, etc.
E.g.
1001 -> 01112 (every idx, except when other pattern goes) x 11 (every idx after end of other pattern) = 011123444567778???

2. Deep cumulative sums
010101 0 -> can decide how long a base pattern we want, e.g. 0-0-0-... or 1-1-1-... is 1, 01-01-... is 2, 101-101-101-... is 3, etc., and we can eliminate prefixes
011223 3
012469 12
0137 13 21 33 ^ can layer as deep as we want to increase complexity
etc.


'''


# Templates
'''

CYCLE GENERALIZATION -> good template
Get all paths given n steps of 3 actions (stay, forward, backward)
Apply them to different starting points to get grid

BASIC RULE (very similar to cycle generalization)
cur_row dependence: 
cur_col dependence: None
actions: stay, forward, backward

start_pos = x
for cur_col in num_cols:
    action

-----
BASIC RULE 2 -> good template
can vary action based on [cur_row, cur_col, cur_pos] [<, >, ==] [cur_row, cur_col, cur_pos, a_len]

start_pos = x
state = T
for cur_col in num_cols:
    if conditionant operation condioner:
        state = !state
    
    if state:
        action1
    else:
        action2

(still needs filtering)
- final column is uniform for last rows
- last rows match (beyond some threshold)
- 1 number takes up ~half of total characters
- looping within rows, i.e. first half etc. of row matches final half of the row

-----
Ideal is a simple generator with interesting patterns. 
That allows for the "AHA!" moment where you realize the simple correct underlying pattern leading to the observations

01302
12413
23024
34130
4024(1)

something that unfolds in interesting ways

recursion?
e.g. the collatz sequence

what about something like taking Game of Life actions and using histories of objects?

-----
BASIC RULE 3
Same as 2, except now searching over state masks that are patterned
lines, fills, etc.
-> what kinds of masks make sense?
1. lines. var number of lines, var number of rotation of the lines, etc.
2. fills
3. classic rules
4. checkerboard
5. etc.
(layering of masks might be something that could take this next next level)

------

LAYERING MASKS

Let each mask be binary.

Say we want to layer a mask. Then, we have one top mask, and two bottom mask, submask1 and submask2

The final mask is the composition of these masks, hence:
- Wherever the top mask is 1, we take the value of submask1
- Wherever the top mask is 0, we take the value of submask2

--> doesn't introduce particularly interesting patterns in general






'''


program_template = '''n = num_rows
l = num_cols
a_len = len(A)
walker = Walker(ordered_alphabet=A, num_rows=n, num_cols=l)

for instance in range(n):
    state = {state_program}
    walker.pos = {walker_start_program}
    
    for time_step in range(l):
        {inner_loop_program}
        
        print(walker.print())'''


class Walker:
    def __init__(self, 
                 ordered_alphabet: List[str],
                 num_rows: int,
                 num_cols: int) -> None:
        self.loop = ordered_alphabet
        self.loop_len = len(ordered_alphabet)
        self.n = num_rows
        self.l = num_cols
        self.pos = None
    
    def set_pos(self, val) -> None:
        self.pos = val % self.loop_len
    
    def forward(self) -> None:
        self.pos = (self.pos + 1) % self.loop_len
    
    def backward(self) -> None:
        self.pos = (self.pos - 1) % self.loop_len
    
    def stay(self) -> None:
        self.pos = self.pos
    
    def print(self) -> str:
        return self.loop[self.pos]
    
    def get_attrs(self) -> dict:
        return {
            'loop': self.loop,
            'loop_len': self.loop_len,
            'n': self.n,
            'l': self.l,
            'pos': self.pos
        }
    

class RuleParam:
    def __init__(self) -> None:
        self.state_program = None
        self.walker_start_program = None
        self.inner_loop_program = None
        self.state_mask = None


class BasicRule(RuleParam):
    def __init__(self, setting, action) -> None:
        super().__init__()

        self.state_program = '''state = True'''
        self.walker_start_program = f'''walker.set_pos({setting})'''
        self.inner_loop_program = f'''{action}'''


class BasicRule2(RuleParam):
    def __init__(self, 
                 setting,
                 conditionant,
                 operation,
                 conditioner,
                 action1,
                 action2) -> None:
        super().__init__()

        self.state_program = '''state = True''' # TODO: Make this parameterizable too
        self.walker_start_program = f'''walker.set_pos({setting})'''
        self.inner_loop_program = f'''if {conditionant} {operation} {conditioner}:
    state = not state

if state:
    {action1}
else:
    {action2}'''
    
    def get_full_program(self):
        return '\n'.join([self.state_program, self.walker_start_program, self.inner_loop_program])


class ConstantRow(RuleParam):
    def __init__(self) -> None:
        super().__init__()

        self.state_program = '''state = True'''
        self.walker_start_program = '''walker.set_pos(instance)'''
        self.inner_loop_program = '''walker.stay()'''


class ConstantCol(RuleParam):
    def __init__(self) -> None:
        super().__init__()

        self.state_program = '''state = True'''
        self.walker_start_program = '''walker.set_pos(-1)'''
        self.inner_loop_program = '''walker.forward()'''


class Cycle(RuleParam):
    def __init__(self) -> None:
        super().__init__()

        self.state_program = '''state = True'''
        self.walker_start_program = '''walker.set_pos(instance - 1)'''
        self.inner_loop_program = '''walker.forward()'''


class Diagonals(RuleParam):
    def __init__(self) -> None:
        super().__init__()

        self.state_program = '''state = True'''
        self.walker_start_program = '''walker.set_pos(instance - 1)'''
        self.inner_loop_program = '''
if walker.pos == a_len - 1:
    state = not state

if state:
    walker.forward()
else:
    walker.backward()
'''


class BackForth(RuleParam):
    def __init__(self) -> None:
        super().__init__()

        self.state_program = '''state = True'''
        self.walker_start_program = '''walker.set_pos(instance - 1)'''
        self.inner_loop_program = '''
if time_step == a_len - 1:
    state = not state

if state:
    walker.forward()
else:
    walker.backward()
'''


class MaskRule(RuleParam):
    def __init__(self, 
                 state_mask,
                 setting,
                 actions,
                 a_len) -> None:
        super().__init__()

        self.state_program = '''state = 0'''
        self.walker_start_program = f'''walker.set_pos({setting})'''

        self.state_mask = state_mask
        self.inner_loop_program = f'''
if state % {len(actions)} == 0:
    {actions[0]}
'''
        for i, action in enumerate(actions[1:]):
            self.inner_loop_program += f'''elif state % {len(actions)} == {i + 1}:
    {action}
'''

    def get_full_program(self):
        return '\n'.join([self.state_program, self.walker_start_program, self.inner_loop_program, '\n', str(self.state_mask)])


walker_class_str = '''from typing import List

class Walker:
    def __init__(self, 
                 ordered_alphabet: List[str],
                 num_rows: int,
                 num_cols: int) -> None:
        self.loop = ordered_alphabet
        self.loop_len = len(ordered_alphabet)
        self.n = num_rows
        self.l = num_cols
        self.pos = None
    
    def set_pos(self, val) -> None:
        self.pos = val % self.loop_len
    
    def forward(self) -> None:
        self.pos = (self.pos + 1) % self.loop_len
    
    def backward(self) -> None:
        self.pos = (self.pos - 1) % self.loop_len
    
    def stay(self) -> None:
        self.pos = self.pos
    
    def print(self) -> str:
        return self.loop[self.pos]
    
    def get_attrs(self) -> dict:
        return {
            'loop': self.loop,
            'loop_len': self.loop_len,
            'n': self.n,
            'l': self.l,
            'pos': self.pos
        }

'''


def get_var_dict_and_class(instance, n, l, ordered_alphabet, state=None, time_step=None, walker=None):
    return {
        'instance': instance,
        'n': n,
        'l': l,
        'ordered_alphabet': ordered_alphabet,
        'a_len': len(ordered_alphabet),
        'state': state,
        'time_step': time_step,
        'walker': Walker(ordered_alphabet, n, l) if walker is None else walker
    }, walker_class_str


def evaluate_state_program(state_program, instance, n, l, ordered_alphabet):
    assert 'state = ' == state_program[:len('state = ')]

    vars, _ = get_var_dict_and_class(instance, n, l, ordered_alphabet)
    exec(state_program, vars)

    return vars['state']


def evaluate_walker_start_program(walker_start_program, instance, n, l, ordered_alphabet, state):
    assert 'walker.set_pos(' == walker_start_program[:len('walker.set_pos(')]

    vars, class_str = get_var_dict_and_class(instance, n, l, ordered_alphabet, state)
    exec(class_str + walker_start_program, vars)

    return vars['walker'].pos


def evaluate_inner_loop_program(inner_loop_program, instance, n, l, ordered_alphabet, state, time_step, walker):
    vars, class_str = get_var_dict_and_class(instance, n, l, ordered_alphabet, state, time_step, walker)
    exec(class_str + inner_loop_program, vars)

    return vars['walker'].pos, vars['state']


def generate_grid(num_rows: int, 
                  num_cols: int, 
                  A: List[str],
                  rule: RuleParam = None,
                  state_program: str = None,
                  walker_start_program: str = None,
                  inner_loop_program: str = None,
                  state_mask: List[List[int]] = None):
    if state_program is None and walker_start_program is None and inner_loop_program is None and rule is not None:
        state_program = rule.state_program
        walker_start_program = rule.walker_start_program
        inner_loop_program = rule.inner_loop_program
        state_mask = rule.state_mask
    elif rule is None and not (state_program is not None and walker_start_program is not None and inner_loop_program is not None):
        raise ValueError('Either only rule must be specified, only all of the program parameters.')
    
    grid_output = []

    n = num_rows
    l = num_cols
    walker = Walker(ordered_alphabet=A, num_rows=n, num_cols=l)

    for instance in range(n):
        # TODO: Account for state properly. Current implementation misses the last state value of each row (p sure, check)
        state = evaluate_state_program(state_program, instance, n, l, A) if state_mask is None else state_mask[instance][0]
        walker.pos = evaluate_walker_start_program(walker_start_program, instance, n, l, A, state)
        
        instance_output = []
        for time_step in range(l):
            instance_output.append(walker.print())
            walker.pos, state = evaluate_inner_loop_program(inner_loop_program, instance, n, l, A, state, time_step, walker)
            if state_mask is not None and time_step < l - 1:
                state = state_mask[instance][time_step + 1]
            
            # print(walker.pos, state)
        
        grid_output.append(instance_output)
        # print('instance output', instance_output)

    return grid_output


def get_rid_of_duplicate_grids(input_list, key=None):
    output_list = []
    key_list = []
    if key is not None:
        for item in input_list:
            if item[key] not in key_list:
                output_list.append(item)
                key_list.append(item[key])
    else:
        for item in input_list:
            if item not in output_list:
                output_list.append(item)
    
    return output_list


def check_frays(grid_list):
    prefixes = {}
    count_frays = 0
    example_frays = {}
    for grid in grid_list:
        key = tuple(list(chain.from_iterable(grid))[:-1])
        if key not in prefixes:
            prefixes[key] = grid[-1][-1]
        elif prefixes[key] != grid[-1][-1]:
            count_frays += 1
            example_frays[key] = [*prefixes[key], grid[-1][-1]]
    
    return count_frays, example_frays


def get_rid_of_frays(input_list, fray_exs, key=None):
    output_list = []
    key_list = []
    if key is not None:
        for item in input_list:
            gkey = tuple(list(chain.from_iterable(item[key]))[:-1])
            if gkey not in fray_exs:
                output_list.append(item)
    else:
        for item in input_list:
            gkey = tuple(list(chain.from_iterable(item))[:-1])
            if gkey not in fray_exs:
                output_list.append(item)
    
    return output_list


def basic_rule_2_generation(n=3, l=3, a_len=3):
    settings = [
        'instance',
        '-instance',
        0
    ]
    conditionants = [
        'time_step',
        'walker.pos',
        'instance'
    ]
    operations = [
        '>',
        '<',
        '=='
    ]
    conditioners = [
        'time_step',
        'walker.pos',
        'instance',
        'a_len',
    ]
    actions = [
        'walker.stay()',
        'walker.forward()',
        'walker.backward()'
    ]
    
    A = [str(elem) for elem in range(a_len)]
    # A = ['A', 'B', 'C', 'D', 'E']
    outputs = []
    for setting in settings:
        for conditionant in conditionants:
            for operation in operations:
                for conditioner in conditioners:
                    for action1 in actions:
                        for action2 in actions:
                            rule = BasicRule2(
                                setting=setting,
                                conditionant=conditionant,
                                operation=operation,
                                conditioner=conditioner,    # TODO: +1, 0, -1 variation
                                action1=action1,
                                action2=action2)
                            grid = generate_grid(
                                num_rows=n,
                                num_cols=l,
                                A=A,
                                rule=rule,
                            )
                            outputs.append([grid, rule.get_full_program()])
    
    outputs = get_rid_of_duplicate_grids(outputs, key=0)
    count_frays, ex_frays = check_frays([item[0] for item in outputs])
    print('EXAMPLES:')
    for i, key in enumerate(ex_frays):
        print('GRID:', key)
        print('ANSWERS:', ex_frays[key])
        print('~~~')
        if i > 10:
            break
    
    outputs = get_rid_of_frays(outputs, ex_frays, key=0)
    
    for item in outputs:
        grid = item[0]
        program = item[1]
        for i, row in enumerate(grid):
            print(', '.join(row if i != len(grid) - 1 else row[:-1] + ['?']))

        print('\n\n\n\n\n\n\n\n', grid[-1][-1])
        print(program)
        print('----------')
    print('NUM FRAYS', count_frays, '/', len(outputs))


def basic_rule_generation(n=3, l=3, a_len=3):
    settings = [
        'instance',
        '-instance',
        0
    ]
    actions = [
        'walker.stay()',
        'walker.forward()',
        'walker.backward()'
    ]

    A = [str(elem) for elem in range(a_len)]
    for setting in settings:
        for action in actions:
            rule = BasicRule(setting=setting, action=action)
            generate_grid(
                num_rows=n,
                num_cols=l,
                A=A,
                rule=rule,
            )
            print('----------')


def test_classic_rules():
    rule_types = [
        ConstantRow(),
        ConstantCol(),
        Cycle(),
        Diagonals(),
        BackForth()
    ]
    
    for rule_type in rule_types:
        generate_grid(
            num_rows=3,
            num_cols=5,
            A=A,
            rule=rule_type,
        )
        print('----------')


def test_single_instance(state_program, walker_start_program, inner_loop_program, n=3, l=3, a_len=3):
    A = [str(elem) for elem in range(a_len)]

    grid = generate_grid(
        num_rows=n,
        num_cols=l,
        A=A,
        state_program=state_program,
        walker_start_program=walker_start_program,
        inner_loop_program=inner_loop_program
    )
    print('----------')
    for row in grid:
        print(row)
    print('----------')


def binary_rule(grid1, grid2, a_len, rule_type='equals'):
    threshold = a_len / 2
    grid3 = []
    for row1, row2 in zip(grid1, grid2):
        row3 = []
        for elem1, elem2 in zip(row1, row2):
            if rule_type == 'equals':
                row3.append(1 if elem1 == elem2 else 0)
            elif rule_type == 'xor':
                row3.append(1 if int(elem1) >= threshold and int(elem2) < threshold or int(elem1) < threshold and int(elem2) >= threshold else 0)
            elif rule_type == 'or':
                row3.append(1 if int(elem1) >= threshold or int(elem2) >= threshold else 0)
            elif rule_type == 'and':
                row3.append(1 if int(elem1) >= threshold and int(elem2) >= threshold else 0)
            elif rule_type == 'and_xor_combo':
                if int(elem1) >= threshold and int(elem2) >= threshold:
                    row3.append(2)
                elif int(elem1) >= threshold and int(elem2) < threshold or int(elem1) < threshold and int(elem2) >= threshold:
                    row3.append(1)
                else:
                    row3.append(0)
            elif rule_type == 'sum_segment':
                elem_sum = int(elem1) + int(elem2)
                to_append = None
                if elem_sum < 2:
                    to_append = 0
                elif elem_sum < 3:
                    to_append = 1
                else:
                    to_append = 2
                row3.append(to_append)

        
        grid3.append(row3)
    
    for row in grid3:
        print(row)
    print('---')

    for row1, row2, row3 in zip(grid1, grid2, grid3):
        formatted_str = []
        for elem1, elem2, elem3 in zip(row1, row2, row3):
            formatted_str.append(f'({elem1}, {elem2}, {elem3})')
        
        print(', '.join(formatted_str))


def test_binary_rule(rule_type='or'):
    n = 3
    l = 3
    A = [str(i) for i in range(l)]
    rule1 = ConstantCol()
    rule2 = Cycle()

    grid1 = generate_grid(
        num_rows=n,
        num_cols=l,
        A=A,
        rule=rule1
    )

    grid2 = generate_grid(
        num_rows=n,
        num_cols=l,
        A=A,
        rule=rule2
    )

    binary_rule(grid1, grid2, len(A), rule_type=rule_type)


def cycle_generalization_rules(n=5, l=5):
    # Generalization of Cycle rule

    all_paths = []
    def generate_paths(lim_n, it, path):
        if it == lim_n:
            all_paths.append(path)
        else:
            for elem in [-1, 0, 1]:
                generate_paths(lim_n, it + 1, path + [elem])

    A = [str(i) for i in range(l)]
    a_len = len(A)
    generate_paths(l - 1, 0, [0])

    absolute_paths = [list(np.cumsum(item)) for item in all_paths]
    # ppr.pprint(absolute_paths)

    grids = []
    for path in absolute_paths:
        row_starts = random.sample([i for i in range(n)], k=n)
        grid = []
        for row_start in row_starts:
            grid.append([A[(row_start + path_pos) % l] for path_pos in path])
        
        grids.append(grid)
    
    for grid in grids:
        for row in grid:
            print(', '.join(row))
        print('-------')


def generate_masks(n, l):
    #  = [[[int() for j in range(l)] for i in range(n)] for it in range(l)]
    assert n == l
    assert l % 2 == 1
    mid = int(l/2)
    # diagonals = [[[int(i - it <= j <= i + it) for j in range(l)] for i in range(n)] for it in range(mid)]
    # diagonals += [[[int(l - 1 - it <= j + i <= l - 1 + it) for j in range(l)] for i in range(n)] for it in range(mid)]
    # checkerboards = [[[int((j + i + it) % 2) for j in range(l)] for i in range(n)] for it in range(1)]
    cross_fills = [[[int(j % l >= i + it) for j in range(l)] for i in range(n)] for it in range(0, mid, 1)]
    # staircase = [[[int(((-1) ** i) * j <= ((-1) ** i) * mid) for j in range(l)] for i in range(n)] for it in range(1)]

    # Too symmetrical
    # v_fills = [[[int(j >= mid - it) for j in range(l)] for i in range(n)] for it in range(-1, 2)]
    # h_fills = [[[int(i >= mid - it) for j in range(l)] for i in range(n)] for it in range(-1, 2)]
    # h_stripes = [[[int(i % 2 == it) for j in range(l)] for i in range(n)] for it in range(1)]
    # v_stripes = [[[int(j % 2 == it) for j in range(l)] for i in range(n)] for it in range(1)]
    # diamond = [[[int(mid - i <= j <= mid + i and mid - (l - 1 - i) <= j <= mid + (l - 1 - i)) for j in range(l)] for i in range(n)] for it in range(1)]

    # TODO: Other patterns: diagonal stripes, classic rules, other stuff. 
    # TODO: Remember, it should ideally be non-symmetrical or copied on both x and y axes!!

    all_masks = []
    # all_masks += diagonals + checkerboards + cross_fills + staircase
    all_masks += cross_fills
    inverses = []

    # ppr.pprint(all_masks)

    for mask in all_masks:
        inverses.append([[int(not bool(mask[i][j])) for j in range(l)] for i in range(n)])

    return all_masks + inverses


def mask_based_rules(n=11, l=11, mask_depth=2):
    A = [str(i) for i in range(l)]
    # A = ['A', 'B', 'C', 'D', 'E']
    a_len = len(A)

    settings = [
        'instance',
        '-instance',
        0 # --> Cannot use this with symmetrical things, i.e. where the final row is found in some other place in the grid
    ]
    base_actions = [
        'walker.stay()',
        'walker.forward()',
        'walker.backward()'
    ]
    state_masks = generate_masks(n, l)
    print('num highest level state masks:', len(state_masks))
    
    # Remember how to do the mask depth. This was just running a machine on the previous mask, right?
    # E.g. mask0 -> machine -> path1 = mask1 -> machine -> path2 = grid
    # TODO: If first two work, see how multiple depth masks work
    actions_seqs = []

    def compute_actions_seqs(remaining_actions, seq):
        if len(remaining_actions) == 0:
            actions_seqs.append(seq)
        else:
            for idx in range(len(remaining_actions)):
                compute_actions_seqs(remaining_actions[:idx] + remaining_actions[idx+1:], seq + [remaining_actions[idx]])

    compute_actions_seqs(base_actions, [])
    print('num action seqs:', len(actions_seqs))

    prev_outputs = state_masks
    outputs = []

    # TODO: Clean this up, and make it able to track multiple levels of programs (i.e. on second iteration, be able to see what the initial mask+program was)
    # TODO: Note down a couple better heuristics. Maybe ok to have simple generations in earlier layers in depth > 0?
    # TODO: See if difficulty can scale a little more smoothly
    # TODO: See if answers are well determined. Play around
    # TODO: E.g. Play around with what base masks are used. E.g. Play around with how actions are assigned based on mask elem number [0, 2, 4] vs [0, 1, 2]
    # TODO: See how models do
    # TODO: See how humans do
    for i in range(mask_depth):
        print('>>', len(prev_outputs))
        temp_outputs = []
        for setting in settings:
            for state_mask in prev_outputs:
                if state_mask[-1] in state_mask[:-1] and setting == 0:  # Heuristic: avoid when final row shows up elsewhere in grid
                    continue

                state_mask = [[int(elem) for elem in row] for row in state_mask]

                for actions_sequence in actions_seqs:
                    # print(len(state_mask), len(state_mask[0]))
                    rule = MaskRule(
                        state_mask=state_mask,
                        setting=setting,
                        actions=actions_sequence,
                        a_len=a_len
                    )
                    grid = generate_grid(
                        num_rows=n,
                        num_cols=l,
                        A=A,
                        rule=rule,
                    )
                    if (grid[-1] in grid[:-1] # Heuristic: avoid when final row shows up elsewhere in grid
                        or all(grid[-1][-1] not in row for row in grid) # H: Avoid when answer character is unused
                        or all(grid[-1][-1] == elem for elem in grid[-1])): # H: Avoid when last row is constant
                        print('skip')
                        ppr.pprint(grid)
                        continue
                    else:
                        print('hit!')
                    
                    outputs.append([grid, rule.get_full_program()])
                    temp_outputs.append(grid)
        
        prev_outputs = temp_outputs

    outputs = get_rid_of_duplicate_grids(outputs, key=0)
    count_frays, ex_frays = check_frays([item[0] for item in outputs])
    print('EXAMPLES:')
    for i, key in enumerate(ex_frays):
        print('GRID:', key)
        print('ANSWERS:', ex_frays[key])
        print('~~~')
        if i > 10:
            break
    
    outputs = get_rid_of_frays(outputs, ex_frays, key=0)
    
    for item in outputs:
        grid = item[0]
        program = item[1]
        for i, row in enumerate(grid):
            print(', '.join(row if i != len(grid) - 1 else row[:-1] + ['?']))

        print('\n\n\n\n\n\n\n\n', grid[-1][-1])
        print(program)
        print('----------')
    print('NUM FRAYS', count_frays, '/', len(outputs))


def main():
    mask_based_rules()

    # cycle_generalization_rules()

    # binary_rule_types = [
    #     # 'equals',
    #     # 'xor',
    #     # 'or',
    #     # 'and',
    #     'and_xor_combo',
    #     'sum_segment'
    # ]

    # for rule in binary_rule_types:
    #     test_binary_rule(rule_type=rule)
    #     print('----------')

    # basic_rule_2_generation(n=3, l=10, a_len=10)

#     state_program = '''state = True'''
#     walker_start_program = '''walker.set_pos(0)'''
#     inner_loop_program = '''
# if walker.pos < instance:
#     state = not state
#     print('>', walker.pos, instance)

# if state:
#     walker.forward()
# else:
#     walker.stay()
# '''
#     test_single_instance(state_program, 
#                          walker_start_program, 
#                          inner_loop_program,
#                          n=5,
#                          l=5,
#                          a_len=5)
    

'''
some notes:
- generation is alright, not bad
- this kind of templated rule creation probably can go fairly far, if I can find some good templates
- the larger the grid, the more complexity I can plausibly support

- with basic2, common boring pattern is 4 rows the same, latter 4 often

- doing well with just the sequence means doing better often

- can do more hand engineering, but not so scalable

- PCFG is definitely worth building up


pruning techniques
- repeated last ~3 rows
- repeated final column value, last ~3 rows
(- distribution/count of various numbers)

other complexities
- compound dependence: easy: xor on 2 attributes

'''


if __name__ == '__main__':
    main()
