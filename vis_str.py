import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import random
from transformers import AutoTokenizer

DEFAULT_ALPHABET = tuple([chr(65 + i) for i in range(26)])
COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'tab:brown', 'tab:gray', 'tab:orange', 'tab:pink', 'lawngreen', 'yellow']
CHAR_TO_COLOR = {letter: color for color, letter in zip(COLORS, DEFAULT_ALPHABET)}
chat_model_directories = [
    # "/data/public_models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct",  # Meta llama models, gen 3, 2
    # "/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    # "/data/public_models/huggingface/meta-llama/Llama-2-13b-chat-hf",
    # "/data/public_models/huggingface/meta-llama/Llama-2-70b-chat-hf",
    # "/data/public_models/huggingface/meta-llama/Llama-2-7b-chat-hf",
    # "/data/public_models/huggingface/mistralai/Mistral-7B-Instruct-v0.3",  # Mistral models, gen latest
    # "/data/public_models/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "/data/public_models/huggingface/mistralai/Mixtral-8x22B-Instruct-v0.1",        # Skipping for now, tokenizer failing to load
    # "/data/public_models/huggingface/Qwen/Qwen1.5-0.5B-Chat",  # Qwen models, gen 2, a few from 1.5
    # "/data/public_models/huggingface/Qwen/Qwen1.5-1.8B-Chat",
    # "/data/public_models/huggingface/Qwen/Qwen1.5-4B-Chat",
    # "/data/public_models/huggingface/Qwen/Qwen2-0.5B-Instruct",
    # "/data/public_models/huggingface/Qwen/Qwen2-1.5B-Instruct",
    # "/data/public_models/huggingface/Qwen/Qwen2-7B-Instruct",
    # "/data/public_models/huggingface/Qwen/Qwen2-72B-Instruct",
    # "/data/public_models/huggingface/tiiuae/falcon-7b-instruct",  # Tiiuae (Falcon) models, gen latest # Using hardcoded chat template
    # "/data/public_models/huggingface/tiiuae/falcon-40b-instruct",   # Using hardcoded chat template
    # "/data/public_models/huggingface/tiiuae/falcon-180B-chat",
    # "/data/public_models/huggingface/deepseek-ai/deepseek-llm-7b-chat",  # Deepseek models, gen latest-1
    # "/data/public_models/huggingface/deepseek-ai/deepseek-llm-67b-chat",
    # "/data/public_models/huggingface/google/gemma-1.1-2b-it",  # Google (Gemma) models, gen 1.1
    # "/data/public_models/huggingface/google/gemma-1.1-7b-it",
    # "/data/public_models/huggingface/01-ai/Yi-6B-Chat",  # 01-ai (Yi) models, gen 1
    # "/data/public_models/huggingface/01-ai/Yi-34B-Chat",
    "/data/sudharsan_sundar/downloaded_models/gemma-2-9b-it"
]


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


def visualize_tokenization(text, model_path):
    # Load the tokenizer
    model_name = model_path.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    print('MODEL:', model_name)
    print('TOKENS:', '|'.join(tokens))
    print('-'*100)
    # token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # # Plot the tokens and their IDs
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.barh(range(len(tokens)), token_ids, align='center', color='skyblue')
    # ax.set_yticks(range(len(tokens)))
    # ax.set_yticklabels(tokens)
    # ax.invert_yaxis()  # Invert y-axis to have the first token on top
    # ax.set_xlabel('Token IDs')
    # ax.set_title(f'Tokenization of "{text}" using {model_name}')
    
    # # Add text labels for token IDs
    # for i, v in enumerate(token_ids):
    #     ax.text(v + 0.2, i, str(v), color='blue', va='center')
    
    # plt.savefig(f'./tokenization_vis/tokenization_{model_name}.png', dpi=300)
    # plt.close()


def main():
    # visualize_problem([[["B"], ["A"], ["C"]], [["B"], ["A"], ["C"]], [["B"], ["A"], ["C"]]])
    # visualize_problem([[["C"], ["B"], ["A"]], [["A"], ["C"], ["B"]], [["B"], ["A"], ["C"]]])
    # visualize_problem([[["A", "F", "I", "L"], ["B", "E", "G", "L"], ["C", "D", "H", "L"]], [["A", "E", "H", "J"], ["B", "D", "I", "J"], ["C", "F", "G", "J"]], [["A", "D", "G", "K"], ["B", "F", "H", "K"], ["C", "E", "I", "K"]]])
    # visualize_problem([[["A", "F", "I", "L"], ["C", "F", "G", "K"], ["B", "F", "H", "J"]], [["A", "D", "H", "L"], ["C", "D", "I", "K"], ["B", "D", "G", "J"]], [["A", "E", "G", "L"], ["C", "E", "H", "K"], ["B", "E", "I", "J"]]])
    # visualize_problem([[["A", "D", "H", "K", "N", "P"], ["B", "D", "G", "K", "O", "R"], ["C", "D", "I", "K", "M", "Q"]], [["B", "E", "H", "J", "N", "P"], ["C", "E", "G", "J", "O", "R"], ["A", "E", "I", "J", "M", "Q"]], [["C", "F", "H", "L", "N", "P"], ["A", "F", "G", "L", "O", "R"], ["B", "F", "I", "L", "M", "Q"]]])
    test_problem1 = '''Consider the following pattern. Each tuple (*) represents (shape type).\nRow 1: (C), (B), (A)\nRow 2: (A), (C), (B)\nRow 3: (B), (A), (?)\n\nPlease determine the correct values for the final tuple of Row 3, (?), which completes the pattern. Please clearly state your final answer as \"The final answer is: [your final answer].\"'''
    test_problem2 = '''Row 1: (C), (B), (A)\nRow 2: (A), (C), (B)\nRow 3: (B), (A), (?)\n\nPlease'''
    test_problem3 = '''Consider the following pattern. Each tuple (_, _, _, _, _, _, _, _, _, _, _, _, _, _) represents (inner shape texture, inner shape orientation, shape type, shape texture, inner shape type, inner shape count, shape count, shape orientation, shape size, inner shape color, inner shape size, inner shape position, shape position, shape color).\nRow 1: (A, F, G, K, M, R, S, V, a, e, i, j, m, q), (B, E, G, L, M, Q, S, X, b, d, h, l, n, r), (C, D, G, J, M, P, S, W, c, f, g, k, o, p)\nRow 2: (C, F, I, K, N, P, U, V, a, f, i, l, m, p), (A, E, I, L, N, R, U, X, b, e, h, k, n, q), (B, D, I, J, N, Q, U, W, c, d, g, j, o, r)\nRow 3: (B, F, H, K, O, Q, T, V, a, d, i, k, m, r), (C, E, H, L, O, P, T, X, b, f, h, j, n, p), (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n\nPlease determine the correct values for the final tuple of Row 3, (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?), which completes the pattern. Please clearly state your final answer as \"The final answer is: [your final answer].\"'''
    test_problem4 = '''shape color).\nRow 1: (A, F, G, K, M, R, S, V, a, e, i, j, m, q), (B, E, G, L, M, Q, S, X, b, d, h, l, n, r), (C, D, G, J, M, P, S, W, c, f, g, k, o, p)\nRow 2: (C, F, I, K, N, P, U, V, a, f, i, l, m, p), (A, E, I, L, N, R, U, X, b, e, h, k, n, q), (B, D, I, J, N, Q, U, W, c, d, g, j, o, r)\nRow 3: (B, F, H, K, O, Q, T, V, a, d, i, k, m, r), (C, E, H, L, O, P, T, X, b, f, h, j, n, p), (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n\nPlease determine the correct values for the final tuple of Row 3, (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    for model_path in chat_model_directories:
        visualize_tokenization(test_problem3, model_path)

    '''
    Notes:
    > problem 2
    - llama 3 is fine, except (?) broken sticky
    - llama 2 is fine, same as mistral, except (?) broken sticky
    - qwen models are fine, except (?) broken sticky
    - deepseek, (?) solid
    - falcon (?) solid
    - gemma models are more or less the same, but (?) is solid, rather than split. nothing special
    - yi has ? correct
    - all have _(, \n(, and ), generally

    > problem 3
    - llama 3: first is single letter, rest are space letter no comma. _(? first and _?, after
    - llama 2: first is single letter, rest are space letter no comma. ?, first and _? after, comma is separate (better)
    - mistral: same as llama 2
    - qwen (1.5 and 2): first is single letter, rest are space letter no comma. _(? first and _?, after
    - falcon: first is single letter, rest are space letter no comma. All are ?, and spaces singled
    - deepseek: first is single letter, rest are space letter no comma. first is ?, rest are _?,
    - gemma (1.1 and 2): first is single letter, rest are space letter no comma. first is ?, rest are _?,
    - yi: all are single letters. ? are all single as well
    '''


if __name__ == '__main__':
    main()
