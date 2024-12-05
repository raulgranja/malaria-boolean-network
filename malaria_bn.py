!pip install python-docx
!pip install matplotlib
!pip install dataframe_image

import numpy as np
from graphviz import Digraph
from itertools import product
import pandas as pd
from google.colab import drive
from docx import Document
import matplotlib.pyplot as plt
import dataframe_image as dfi
from IPython.display import Image
import random
from collections import Counter

def sort_row(row):
    # Extract the ID
    oligo_id = row['Oligo_ID']

    # Extract the name
    name = row['NAME']

    # Sort the numeric values starting from the third column
    sorted_values = sorted(row[2:])

    # Return a new row with 'Oligo_ID', 'NAME', and the sorted numeric values
    return [oligo_id, name] + sorted_values


def list_num(list_s):
    """
    Converts a binary list into the corresponding decimal number.
    Example: [0, 0, 0, 1] --> 1 * 2^0 = 1
             [1, 0, 0, 1] --> 1 * 2^3 + 0 + 0 + 1 * 2^0 = 9

    :param list_s: A list of binary digits (0s and 1s).
    :return: Decimal number (int).
    """
    # Use the built-in int function with base 2 to convert binary to decimal
    return int("".join(map(str, list_s)), 2)


def check_zero(state):
    """
    Identifies which indexes have a value of 1 in the given state and returns them in a list.

    :param state: A list representing the state to analyze.
    :return: A list of indexes where the elements are equal to 1.

    Example:
    - check_zero([0, 1, 0, 1, 1, 1, 0, 0, 0]) returns [1, 3, 4, 5].
    """
    # Use list comprehension to get the indexes where the value is 1
    ones_index = [i for i, elem in enumerate(state) if elem == 1]

    return ones_index


def sum_values(state, attempt):
    """
    Sums the values from 'attempt' at the indices where 'state' has 1s.

    :param state: A list representing the state to analyze.
    :param attempt: A list with values corresponding to the indices in the state.
    :return: Sum of the values in 'attempt' at the indices where 'state' has 1s.

    Example:
    - sum_values([0, 1, 0, 1, 1, 1, 0, 0, 0], [-1, -1, -1, 0, 1, 1, 0, 1, 0]) returns 1.
    """
    # Use sum with a list comprehension to add values of attempt at indices from check_zero(state)
    return sum(attempt[i] for i in check_zero(state))


def smallest_lines(input_file, output_file, gap=0):
    """
    Filters the lines of a given file, keeping only those that have the highest number of zeros within a given range.

    :param input_file: File to be filtered.
    :param output_file: The file where the filtered lines will be written.
    :param gap: The allowed range from the maximum number of zeros (default is 0).
    :return: output_file with the filtered lines.
    """
    # Read the input file and determine the maximum number of zeros
    with open(input_file, 'r') as file1:
        lines = file1.readlines()
        lm = max(line.count('0') for line in lines)

    # Filter and write lines with zeros count within the gap range
    with open(output_file, 'w') as file2:
        for line in lines:
            zeros = line.count('0')
            if zeros >= lm - gap:
                file2.write(line)

    return output_file


def remove_repeated_rows(np_array):
    """
    Removes repeated rows from a 2D NumPy array, retaining only the first occurrence of each unique row
    while preserving the original order of the rows.

    :param np_array: A 2D NumPy array from which repeated rows will be removed.
    :return: A new 2D NumPy array containing only the unique rows, in their original order.
    """
    # Identify unique rows and their original indices
    unique_array, unique_indices = np.unique(np_array, axis=0, return_index=True)

    # Sort the unique indices to maintain the original order of rows
    sorted_unique_indices = np.sort(unique_indices)

    # Extract the unique rows in their original order
    new_array = np_array[sorted_unique_indices]

    return new_array

def readFile(fileName):
    """
    Reads a file that specifies the possibilities of a line and returns a list with integer values.

    :param fileName: The file that will be read.
    :return: List containing all the lines with integer values.
    """
    lines_new = []

    # Open the file using 'with' to ensure it's properly closed
    with open(fileName, "r") as file_obj:
        # Read and process each line
        for line in file_obj:
            line_new = line.strip()
            line_new = [int(i) for i in line_new.split()]
            lines_new.append(line_new)

    return lines_new

def random_reg(n):
    """
    Chooses a random regulation matrix based on the lines files.

    :param n: The number of lines of the matrix, or the number of functional groups, or the number of files.
    :return: A random regulation matrix.
    """
    regulation = []

    for i in range(1, n + 1):
        # Read the file for the current functional group
        file_name = f'small_{i:02}.txt'
        try:
            lis = readFile(file_name)
            # Choose a random line from the list
            regulation.append(random.choice(lis))
        except FileNotFoundError:
            print(f"Warning: {file_name} not found.")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")

    return regulation

def draw_trajectory (np_array):
    """
    Draws a directed graph using Graphviz where the vertices represent the rows of a NumPy array,
    and the directed edges represent transitions from one row to the next in sequence. The graph
    is saved as a PNG image and displayed in the notebook.

    :param np_array: A 2D NumPy array where each row represents a vertex, and directed edges are created
                     between consecutive rows.
    :return: None. The function generates and displays a PNG image of the graph.
    """
    dot = Digraph(comment='My Graph')

    for i in range(len(np_array) - 1):
      dot.node(str(np_array[i]))
      dot.node(str(np_array[i+1]))
      dot.edge(str(np_array[i]), str(np_array[i+1]))

    dot.render('my_graph', view=True, format='png')

    # Display the image in the notebook
    from IPython.display import Image
    Image(filename='my_graph.png')


def solutions_02(D):
    """
    Calculates and writes the valid regulatory patterns for each functional group in a Boolean network based on
    the given state transitions matrix. The function iterates through all possible regulatory patterns
    (-1, 0, 1) and checks their validity based on transitions between states.

    :param D: A 2D NumPy array representing the state transitions, where rows are time points and
              columns represent functional groups. Each element is either 0 or 1, indicating functional group states.
    :return: The total number of valid regulatory matrices.
    """
    n_times, n_modules = D.shape
    characters = [-1, 0, 1]
    total = 1
    square = lambda x: x ** 2

    # Generate all possible combinations for regulatory patterns
    attempts = np.array(list(product(characters, repeat=n_modules)))

    for functional_group in range(n_modules):
        valid_regulations = []  # To store valid regulation attempts

        for attempt in attempts:
            # Skip attempts where all elements are zero (i.e., no regulation)
            if np.sum(attempt ** 2) == 0:
                continue

            # Count valid transitions
            valid = True
            for j in range(n_times - 1):
                current_state = D[j]
                next_state = D[j + 1]

                # Calculate h based on current state and regulation attempt
                h = sum_values(current_state, attempt)

                # Check if the regulation attempt satisfies conditions for functional group
                if not (
                    (next_state[functional_group] == 0 and h < 0) or
                    (next_state[functional_group] == 1 and h > 0) or
                    (current_state[functional_group] == next_state[functional_group] and h == 0)
                ):
                    valid = False
                    break  # No need to check further if this attempt fails

            # If the attempt is valid for all transitions, store it
            if valid:
                valid_regulations.append(attempt)

        # Write valid regulations for the current functional group to a file
        filename = f'functional_group_{functional_group + 1:02}.txt'
        with open(filename, 'w') as f:
            for regulation in valid_regulations:
                f.write(' '.join(map(str, regulation)) + '\n')

        # Output functional group information and update total count
        num_solutions = len(valid_regulations)
        print(f'Functional Group {functional_group + 1:02} has {num_solutions} possible regulation patterns')
        total *= num_solutions

    print(f'Total possible matrices = {total:e}')
    return total


def next_state(actual_state, regulation):
    """
    Calculate the next functional group state based on the current state and a regulation matrix.

    :param actual_state: The current state of functional groups (list or array of 0s and 1s).
    :param regulation: A regulation matrix where each row corresponds to how the state of one functional group is regulated by others.
    :return: The next functional group state as a list of 0s and 1s.
    """
    s_old = np.array(actual_state)
    s_new = []

    for row in regulation:
        sm = np.dot(s_old, row)
        if sm > 0:
            s_new.append(1)
        elif sm < 0:
            s_new.append(0)
        else:
            s_new.append(s_old[len(s_new)])

    return s_new


def num_list(int_s):
    """
    Converts a decimal number to binary, separates all digits, and stores them in a list.

    :param int_s: Any integer number from 0 to 2047 (inclusive).
    :return: A binary list of eleven digits corresponding to the binary representation of the number.

    Example:
    - Converts 2 into [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0].
    - Converts 2047 into [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].
    """
    str_s = f'{int_s:012b}'

    list_s = [int(s) for s in str_s]

    return list_s


def next_state_dec(s_dec, R):
    """
    Computes the next state of a functional group regulation system given the current state in decimal form.

    :param s_dec: The current state of the functional group in decimal format.
    :param R: The regulation matrix used to determine the next state.
    :return: The next state of the functional group in decimal format.
    """
    s_bin = num_list(s_dec)

    next_state_bin = next_state(s_bin, R)

    n_state_dec = list_num(next_state_bin)

    return n_state_dec

def generate_STG(R, n=12):
    """
    Generates a State Transition Graph (STG) for a functional group regulation system.

    :param R: The regulation matrix used to determine state transitions.
    :param n: The number of functional groups (or length of each state) in the system. Default is 7.
    :return: A list representing the State Transition Graph, where each index corresponds to a state,
             and the value at that index is the next state in decimal form.
    """
    if n <= 0:
        raise ValueError("The number of functional groups (n) must be a positive integer.")

    g = [0] * (2 ** n)

    for s in range(2 ** n):
        g[s] = next_state_dec(s, R)

    return g


def draw_STG(M, n=12, STG=None, engine='sfdp', bin_state=False):
    """
    Draws a State Transition Graph (STG) based on a transition matrix and state configurations.

    :param M: The transition matrix that defines state transitions.
    :param n: The number of states to consider (default is 12).
    :param STG: A predefined state transition graph (optional). If not provided, it will be generated from M.
    :param engine: The layout engine to use for the graph visualization (default is 'sfdp').
    :param bin_state: Boolean flag to display states in binary (True) or decimal (False) format (default is False).
    :return: A Digraph object representing the state transition graph.
    """
    if STG is None:
        STG = generate_STG(M, n)

    g = Digraph('G', engine=engine)
    g.attr(size='16,10')

    for ini_state, fim_state in enumerate(STG):
        if bin_state:
            ini_str = ''.join(map(str, num_list(ini_state)))
            fim_str = ''.join(map(str, num_list(fim_state)))
        else:
            ini_str = str(ini_state)
            fim_str = str(fim_state)

        g.edge(ini_str, fim_str)

    return g


def component_sizes(STG):
    """
    Calculates the size of each connected component in a State Transition Graph (STG).

    Args:
        STG: A list representing the State Transition Graph, where each index corresponds to a state,
             and the value at that index is the next state in decimal form.

    Returns:
        component_size_counts: A dictionary where keys are component IDs and values are the sizes of each component.
        total_components: The total number of distinct connected components in the graph.
        state_to_component_map: A list where each index represents a state, and the value is the component ID
                                to which the state belongs.
    """
    # Number of states in the STG
    num_states = len(STG)

    # List to track whether a state has been visited
    visited_states = [False] * num_states

    # List to store the component ID for each state
    state_component_ids = [-1] * num_states

    # Initialize the component counter (component ID)
    current_component_id = 0

    # Loop through each state to explore its connected component
    for state in range(num_states):
        if not visited_states[state]:
            # Start a queue to perform BFS and track the component trajectory
            bfs_queue = [state]

            # Mark the current state as visited and assign it a component ID
            visited_states[state] = True
            state_component_ids[state] = current_component_id

            # Store the current trajectory of states in this component
            current_trajectory = []

            # BFS to explore all states in the connected component
            while bfs_queue:
                current_state = bfs_queue.pop(0)  # Get the current state from the queue
                current_trajectory.append(current_state)

                # Get the next state according to the transition graph
                next_state = STG[current_state]

                # If the next state has not been visited, add it to the queue and mark it as visited
                if not visited_states[next_state]:
                    visited_states[next_state] = True
                    bfs_queue.append(next_state)
                    state_component_ids[next_state] = current_component_id
                else:
                    # If we reach an already visited state, update all states in the trajectory
                    for trajectory_state in current_trajectory:
                        state_component_ids[trajectory_state] = state_component_ids[next_state]
                    break

            # If the BFS queue is empty, increment the component ID for the next group of states
            if not bfs_queue:
                current_component_id += 1

    # Get the unique component IDs and map them to a continuous range
    unique_component_ids = sorted(set(state_component_ids))
    component_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_component_ids)}

    # Remap the component IDs to the continuous range
    state_to_component_map = [component_id_mapping[id] for id in state_component_ids]

    # Count the number of states in each component
    component_size_counts = Counter(state_to_component_map)

    return component_size_counts, len(unique_component_ids), state_to_component_map


def generate_random_regulation(num_functional_groups, seed_value=10):
    """
    Generates a random regulation matrix for a given number of functional groups.

    Args:
        num_functional_groups (int): The number of functional groups for which to generate the regulation matrix.
        seed_value (int, optional): The seed for random number generation to ensure reproducibility. Defaults to 10.

    Returns:
        list: A matrix representing the functional group regulation rules where each row indicates how the state of one functional group is influenced by others.
    """
    random.seed(seed_value)
    regulation_matrix = random_reg(num_functional_groups)
    return regulation_matrix


def generate_and_analyze_STG(num_functional_groups=12, seed_value=10):
    """
    Generates a State Transition Graph (STG) for a given number of functional groups, analyzes its connected components,
    and prints the number of distinct connected components and their size frequencies.

    Args:
        num_functional_groups (int, optional): The number of functional groups in the regulation matrix. Defaults to 12.
        seed_value (int, optional): The seed for random number generation to ensure reproducibility. Defaults to 10.

    Returns:
        None: The function prints the results directly.
    """
    regulation_matrix = generate_random_regulation(num_functional_groups, seed_value)
    STG = generate_STG(regulation_matrix, n=num_functional_groups)
    frequency_counts, num_distinct_components, mapped_component_ids = component_sizes(STG)
    print(f"Number of distinct connected components: {num_distinct_components}")
    print("Component size frequencies:", frequency_counts)
    return frequency_counts


def plot_component_frequencies(frequency_counts):
    """
    Plots a bar chart representing the frequency of each connected component in a State Transition Graph (STG),
    excluding components with zero frequency. Each bar shows the number of states within each component.

    Args:
        frequency_counts (dict): A dictionary where keys are component IDs and values are the corresponding frequencies.

    Returns:
        None: Displays a bar chart of the component sizes.
    """
    filtered_counts = {id: count for id, count in frequency_counts.items() if count > 0}
    ids = sorted(filtered_counts.keys())
    frequencies = [filtered_counts[id] for id in ids]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(ids, frequencies, color='skyblue')

    plt.xlabel("Component ID", fontsize=12)
    plt.ylabel("Number of States", fontsize=12)
    plt.title("Number of States in each Connected Component", fontsize=14)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def draw_tbn(adjacency_matrix, module_names):
    """
    Draws a thresholded Boolean network (TBN) based on the provided adjacency matrix.

    Args:
        adjacency_matrix (list of list of float): A square matrix representing the transitions between modules.
        module_names (list of str): A list containing names for each module.

    Returns:
        Digraph: A Graphviz Digraph object representing the TBN.
    """
    graph = Digraph('G', node_attr={'style': 'filled', 'fontname': 'Helvetica', 'fontsize': '12'})
    graph.attr(size='10,7', layout='dot')  # Set layout and size for the graph
    graph.attr(rankdir='TB')  # Top to bottom layout

    num_modules = len(adjacency_matrix)

    # Define colors for positive and negative edges
    positive_color = 'lightblue'
    negative_color = 'salmon'

    # Create nodes with customized colors
    for i, name in enumerate(module_names):
        graph.node(str(name), color=positive_color, shape='ellipse', fontcolor='black', style='filled')

    for i in range(num_modules):
        for j in range(num_modules):
            if adjacency_matrix[i][j] < 0:
                graph.edge(str(module_names[j]), str(module_names[i]), arrowhead="tee", color=negative_color, penwidth='2')
            elif adjacency_matrix[i][j] > 0:
                graph.edge(str(module_names[j]), str(module_names[i]), color=positive_color, penwidth='2')

    return graph

drive.mount('/content/drive')

file_path = 'dados_malaria.csv'

df_original = pd.read_csv(file_path, delimiter = '\t')

cols_to_convert = df_original.columns[2:]

df_original[cols_to_convert] = df_original[cols_to_convert].apply(pd.to_numeric, errors='coerce')

df_original_dropped = pd.concat([df_original[df_original.columns[:2]], df_original[cols_to_convert]], axis=1).dropna(how='any').reset_index(drop=True)

sorted_data = df_original_dropped.apply(sort_row, axis=1).tolist()

columns = df_original_dropped.columns.tolist()

df_sorted = pd.DataFrame(sorted_data, columns=columns)

new_columns_names = {f'TP {i}': f'{i}' for i in range(1, 49)}

df_sorted.rename(columns=new_columns_names, inplace=True)

fixed_columns = df_sorted.iloc[:, :2]

columns_to_diff = df_sorted.iloc[:, 2:]

columns_diff = columns_to_diff.diff(axis=1)

df_diff = pd.concat([fixed_columns, columns_diff], axis=1)

df_diff = df_diff.drop('1', axis=1)

new_columns_name = {f'{i}': f'{i - 1}' for i in range(0, 47)}

df_diff.rename(columns=new_columns_name, inplace=True)

df_sorted_str = df_sorted.iloc[:, :2]
df_sorted_num = df_sorted.iloc[:, 2:]

diff = df_sorted_num.iloc[:, -1] - df_sorted_num.iloc[:, 0]
result = diff / (df_sorted_num.shape[1] - 1)

df_t = df_sorted_str.copy()
df_t['t'] = result

result_data = []

for _, row2 in df_t.iterrows():
    oligo_id = row2['Oligo_ID']
    name_val = row2['NAME']
    t_value = row2['t']

    row1 = df_diff[(df_diff['Oligo_ID'] == oligo_id) & (df_diff['NAME'] == name_val)]

    min_index = None
    for col in df_diff.columns[2:]:
        if row1[col].values[0] > t_value:
            min_index = col
            break

    result_data.append([oligo_id, name_val, min_index])

df_min = pd.DataFrame(result_data, columns=['Oligo_ID', 'NAME', 'MIN_INDEX'])
df_min['MIN_INDEX'] = pd.to_numeric(df_min['MIN_INDEX'], errors='coerce').astype(pd.Int64Dtype())

new_columns_names = {f'{i}': f'TP {i}' for i in range(1, 49)}
df_sorted.rename(columns=new_columns_names, inplace=True)

df_B = pd.concat([df_original_dropped.iloc[:, :2], pd.DataFrame(0, index=df_original_dropped.index, columns=df_original_dropped.columns[2:], dtype=int)], axis=1)

for i in range(len(df_original_dropped)):
    m = df_min.loc[i, 'MIN_INDEX']

    for j in df_original_dropped.columns[2:]:
        m_plus_1 = m + 1

        if (m_plus_1 == 23) or (m_plus_1 == 29):
            m_plus_1 = m + 2
        elif m_plus_1 == 49:
            m_plus_1 = 48

        col = 'TP ' + str(m_plus_1)

        if df_original_dropped.loc[i, j] >= df_sorted.loc[i, col]:
            df_B.loc[i, j] = 1
        else:
            df_B.loc[i, j] = 0

file_path = 'functional_groups_pandas.xlsx'

df_functional_groups = pd.read_excel(file_path)

functional_groups_first_col = df_functional_groups.iloc[:, 0:1]
df_functional_groups_numbers = df_functional_groups.iloc[:, 4:]

df_functional_groups_averages = pd.concat([functional_groups_first_col, df_functional_groups_numbers], axis=1).groupby('functional_group', sort=False).mean().reset_index()

df_functional_groups_averages_transposed = df_functional_groups_averages.transpose()
df_functional_groups_averages_transposed.columns = df_functional_groups_averages_transposed.iloc[0]
df_functional_groups_averages_transposed = df_functional_groups_averages_transposed.drop(df_functional_groups_averages_transposed.index[0])
df_functional_groups_averages_transposed = df_functional_groups_averages_transposed.iloc[:, :-2]

df_binarized = df_functional_groups_averages_transposed.copy()
column_means = df_binarized.mean()

for column in df_binarized.columns:
    df_binarized[column] = (df_binarized[column] >= column_means[column]).astype(int)

df_binarized_np = df_binarized.to_numpy()

# Create a copy of the original array to avoid modifying the original
new_bin_array = df_binarized_np.copy()

# Update specific elements in the array to zero
new_bin_array[5, 3] = 0   # Set element at row 5, column 3 to 0
new_bin_array[3, 3] = 0   # Set element at row 3, column 3 to 0
new_bin_array[3, 10] = 0  # Set element at row 3, column 10 to 0
new_bin_array[12, 7] = 0  # Set element at row 12, column 7 to 0
new_bin_array[18, 7] = 0  # Set element at row 18, column 7 to 0

cleaned_bin_array = remove_repeated_rows(new_bin_array)

# Appending the first row of 'new_array' to the end of 'new_array'
appended_array = np.append(cleaned_bin_array, [cleaned_bin_array[0]], axis=0)

trajetoria = [list_num(list(row)) for row in appended_array]

draw_trajectory(appended_array)
Image(filename='my_graph.png')

solutions_02 (appended_array)

n_times, n_modules = appended_array.shape

for gene in range(n_modules):
    input_file = f'functional_group_{gene + 1:02}.txt'
    output_file = f'small_{gene + 1:02}.txt'

    # Print the filenames for debugging purposes
    print(f'Processing input file: {input_file}, output file: {output_file}')
    smallest_lines(input_file, output_file, gap=0)

n_times, n_modules = appended_array.shape
number_matrices = 1  # Initialize the product for counting matrices

for gene in range(n_modules):
    input_file = f'small_{gene + 1:02}.txt'

    # Efficiently count the number of lines without loading the entire file into memory
    with open(input_file, 'r') as file:
        line_count = sum(1 for _ in file)

    print(f'{input_file} has {line_count:02} rows')
    number_matrices *= line_count  # Multiply to keep track of total possible matrices

print(f'Total possible matrices: {number_matrices}')

frequency_counts = generate_and_analyze_STG(num_functional_groups=12, seed_value=10)

R = generate_random_regulation(12, seed_value=10)
draw_STG(R)

plot_component_frequencies(frequency_counts)

STG = generate_STG(R, n=12)
frequency_counts, num_distinct_ids, mapped_component_ids = component_sizes(STG)

# Convert each row of appended_array from binary to decimal
decimal_values = [int(''.join(map(str, row)), 2) for row in appended_array]

random.seed(10)

# Initialize counters and lists to store results
attempt_count = 0
basin_counts = []
malaria_basin_sizes = []
min_distinct_ids = 4096

while attempt_count < 50:
    regulation_matrix = random_reg(12)
    state_transition_graph = generate_STG(regulation_matrix, n=12)

    # Calculate component sizes from the State Transition Graph
    frequency_counts, num_distinct_ids, mapped_component_ids = component_sizes(state_transition_graph)

    # Get the size of the basin of attraction containing the malaria sequence
    basin_size = frequency_counts[mapped_component_ids[decimal_values[0]]]

    basin_counts.append(num_distinct_ids)
    malaria_basin_sizes.append(basin_size)

    # Update the minimum distinct IDs and associated basin size if necessary
    if num_distinct_ids < min_distinct_ids:
        min_distinct_ids = num_distinct_ids
        current_basin_size = basin_size
        best_regulation_matrix = regulation_matrix

    print(attempt_count)
    attempt_count += 1

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(basin_counts, malaria_basin_sizes, color='skyblue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.xlabel("Number of Basins of Attraction", fontsize=12)
plt.ylabel("Size of Basin of Attraction Containing Malaria Sequence", fontsize=12)
plt.title("Scatter Plot of Number of Basins vs. Basin Size", fontsize=14)

# Add a grid
plt.grid(True, alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()

plot_component_frequencies(frequency_counts)

best_regulation_matrix

print(f"This network has {min_distinct_ids} basins of attraction.")
print(f"The size of the basin containing the malaria cycle is: {current_basin_size}.")

module_names = df_functional_groups_averages_transposed.columns.tolist()
tbn_graph = draw_tbn(best_regulation_matrix, module_names)
tbn_graph.render('tbn_graph', format='png', cleanup=True)  # Save the graph as a PNG file

tbn_graph
