import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import random

DEFAULT_ALPHABET = tuple([chr(65 + i) for i in range(26)])
COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'tab:brown', 'tab:gray', 'tab:orange', 'tab:pink', 'lawngreen', 'yellow']
CHAR_TO_COLOR = {letter: color for color, letter in zip(COLORS, DEFAULT_ALPHABET)}


def visualize_problem(problem_abstraction):
    # Get a list of colors from matplotlib
    colors = list(mcolors.CSS4_COLORS.values())

    # Shuffle the colors to randomize the color assignment
    random.shuffle(colors)

    # Create a dictionary to store character-to-color mappings
    char_to_color = CHAR_TO_COLOR
    char_to_color[' '] = 'black'
    char_to_color['?'] = 'white'

    # Split the string into rows
    problem_abstraction[2][2] = '?' * len(problem_abstraction[2][2])
    rows = ['  '.join([''.join(elem) for elem in row]) for row in problem_abstraction]

    fig, ax = plt.subplots(figsize=(len(rows[0]), len(rows) * 2))
    ax.set_xlim(0, len(rows[0]))
    ax.set_ylim(0, len(rows))

    # Plot each character as a colored rectangle
    for row_idx, row in enumerate(rows):
        for col_idx, char in enumerate(row):
            ax.add_patch(plt.Rectangle((col_idx, len(rows) - row_idx - 1), 1, 1, color=char_to_color[char], ec='white'))
            # ax.text(col_idx + 0.5, len(rows) - row_idx - 1 + 0.5, char, ha='center', va='center', fontsize=12, color='white')

    ax.axis('off')  # Turn off the axis
    plt.show()


def main():
    visualize_problem([[["B"], ["A"], ["C"]], [["B"], ["A"], ["C"]], [["B"], ["A"], ["C"]]])
    visualize_problem([[["C"], ["B"], ["A"]], [["A"], ["C"], ["B"]], [["B"], ["A"], ["C"]]])
    visualize_problem([[["A", "F", "I", "L"], ["B", "E", "G", "L"], ["C", "D", "H", "L"]], [["A", "E", "H", "J"], ["B", "D", "I", "J"], ["C", "F", "G", "J"]], [["A", "D", "G", "K"], ["B", "F", "H", "K"], ["C", "E", "I", "K"]]])
    visualize_problem([[["A", "F", "I", "L"], ["C", "F", "G", "K"], ["B", "F", "H", "J"]], [["A", "D", "H", "L"], ["C", "D", "I", "K"], ["B", "D", "G", "J"]], [["A", "E", "G", "L"], ["C", "E", "H", "K"], ["B", "E", "I", "J"]]])
    visualize_problem([[["A", "D", "H", "K", "N", "P"], ["B", "D", "G", "K", "O", "R"], ["C", "D", "I", "K", "M", "Q"]], [["B", "E", "H", "J", "N", "P"], ["C", "E", "G", "J", "O", "R"], ["A", "E", "I", "J", "M", "Q"]], [["C", "F", "H", "L", "N", "P"], ["A", "F", "G", "L", "O", "R"], ["B", "F", "I", "L", "M", "Q"]]])


if __name__ == '__main__':
    main()
