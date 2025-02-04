import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
import time
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool
import multiprocessing

start = time.time()


def hyperbolic_distance(z1, z2):
    d = np.arccosh(1 + (2 * (abs(z1 - z2)) ** 2 / ((1 - abs(z1) ** 2) * (1 - abs(z2) ** 2))))
    return d


def unit_cell_positions(p, q):
    a = np.pi / p
    b = np.pi / q
    r0 = np.sqrt(np.cos(a + b) / np.cos(a - b))
    vertices_positions = []
    if p == 8 and q == 3:
        for k in range(p):
            vertices_positions.append(r0 * np.exp(1j * np.pi * (2 * k - 1) / p))
        d0 = 0.66
        for k in range(p):
            vertices_positions.append(d0 * np.exp(1j * np.pi * (2 * k - 1) / p))
    else:
        for k in range(p):
            vertices_positions.append(r0 * np.exp(1j * np.pi * (2 * k) / p))
    return vertices_positions


def rotation_matrix(phi):
    return np.array([[np.exp(1j * phi / 2), 0], [0, np.exp(-1j * phi / 2)]])


def fuchsian_generators(p_B, q_B):
    alpha = 2* np.pi/ p_B
    beta = 2* np.pi/ q_B
    sigma = np.sqrt((np.cos(alpha) + np.cos(beta)) / (1 + np.cos(beta)) )
    gamma1 = 1/(np.sqrt(1-sigma**2)) * np.array([[1, sigma], [sigma,1]])
    FG_generators = []
    for mu in range(0,int(p_B/2)):
        gamma_j = rotation_matrix(mu*alpha) @ gamma1 @ rotation_matrix(-mu*alpha)
        FG_generators.append(gamma_j)
        FG_generators.append(np.linalg.inv(gamma_j))
    return FG_generators


def create_new_vertex(vertex_position, translation_matrix):
    v = translation_matrix @ np.array([vertex_position, 1])
    new_vertex_position = v[0] / v[1]
    return new_vertex_position


def generate_hyperbolic_graph(vertices, draw=False):
    vertices_to_edges = {}
    edges_to_vertices = {}
    d0 = hyperbolic_distance(vertices[0], vertices[1])
    coords = [(v.real, v.imag) for v in vertices]

    G = nx.Graph()
    for idx, pos in enumerate(coords):
        G.add_node(idx, pos=pos, label=True)

    edge_count = 0  # Counter for edge labels

    for i, pos1 in enumerate(vertices):
        for j, pos2 in enumerate(vertices[i + 1:], start=i + 1):
            if hyperbolic_distance(pos1, pos2) < (d0 + 0.1):
                G.add_edge(i, j, with_labels=True)
                vertices_to_edges[(i, j)] = edge_count
                edges_to_vertices[edge_count] = (i, j)
                edge_count += 1

    pos_dict = {idx: pos for idx, pos in enumerate(coords)}

    # Set figure size
    # plt.figure(figsize=(20, 20))  # Adjust the width and height as needed
    if draw:
        # Draw the graph
        nx.draw(
            G,
            pos=pos_dict,
            node_size=50,  # Adjust node size
            node_color="lightblue",
            with_labels=True,
            font_size=12,  # Increase font size for node labels
            font_color="black"
        )

        # Draw edge labels
        nx.draw_networkx_edge_labels(
            G,
            pos=pos_dict,
            edge_labels=vertices_to_edges,
            font_size=10,  # Increase font size for edge labels
            label_pos=0.5,  # Adjust edge label position (closer to the center of edges)
        )

        plt.axis("equal")  # Ensure equal scaling
        plt.show()  # Display the plot

    return G, vertices_to_edges, edges_to_vertices


def get_edge_labels_for_vertex(G, vertex, vertices_to_edges):
    # Get all edges incident to the given vertex
    incident_edges = list(G.edges(vertex))
    # Retrieve the labels for these edges from the vertices_to_edges dictionary
    incident_edge_labels = {edge: vertices_to_edges[tuple(sorted(edge))] for edge in incident_edges}
    return incident_edge_labels.values()


def generate_vertices(p,q, p_B, q_B, N):
    unit_cell = unit_cell_positions(p, q)

    d0 = hyperbolic_distance(unit_cell[0], unit_cell[1])

    group_generators = fuchsian_generators(p_B, q_B)

    outer_rings = []

    if N == 1:
        return unit_cell
    elif N == p_B+1:
        extra_generators_indices = []
    elif N == 4+p_B:
        extra_generators_indices = [(2, 0), (3, 1), (6, 3)]
    elif N == 7 + p_B:
        extra_generators_indices = [(2, 0), (3, 1), (6, 3), (5, 2), (7, 2), (6, 6)]


    for generator in group_generators:
        for vertex in unit_cell:
            new_vertex = create_new_vertex(vertex, generator)
            # Check if new_vertex is not in any of the vertices in D
            if all(hyperbolic_distance(new_vertex, vertex) > (d0-0.15) for vertex in unit_cell + outer_rings):
                outer_rings.append(new_vertex)

    for index_pair in extra_generators_indices:
        generator = group_generators[index_pair[0]] @ group_generators[index_pair[1]]
        for vertex in unit_cell:
            new_vertex = create_new_vertex(vertex, generator)
            if all(hyperbolic_distance(new_vertex, vertex) > (d0 - 0.15) for vertex in unit_cell + outer_rings):
                outer_rings.append(new_vertex)

    return unit_cell + outer_rings


def create_adjacency_matrix(N, CT, p_B):
    # Parameters

    unit_cell = unit_cell_positions(p, q)
    n = len(unit_cell)  # Number of vertices in the unit cell
    unit_cell_graph = generate_hyperbolic_graph(unit_cell)

    V = nx.to_numpy_array(unit_cell_graph[0])  # Adjacency matrix of the unit cell
    # Initialize the zero matrices
    T_matrices = [np.zeros((n, n)) for _ in range(p_B)]

    # Indices to be updated
    T_indices = [
        [[9, 12], [8, 13]],
        [[12, 9], [13, 8]],
        [[9, 14], [10, 13]],
        [[14, 9], [13, 10]],
        [[10, 15], [11, 14]],
        [[15, 10], [14, 11]],
        [[11, 8], [12, 15]],
        [[8, 11], [15, 12]]]

    # Update the T matrices
    for j in range(p_B):
        for k, l in T_indices[j]:
            T_matrices[j][k, l] = 1

    # Initialize A_l
    I = np.identity(N)
    A_l = np.kron(I, V)

    # Perform the update to A_l based on CT
    for alpha in range(p_B):  # Iterate over the T_matrices
        for i in range(N):
            # Create U matrix
            U = np.zeros((N, N))
            j = CT[alpha][i] - 1  # Adjust indexing for Python (0-based)
            U[i, j] = 1
            # Update A_l
            A_l += np.kron(U, T_matrices[alpha])

    # Convert A_l to a sparse matrix (if not already sparse)
    A_l_sparse = csr_matrix(A_l)

    # Convert to COO format to access row, col, and data attributes
    A_l_coo = coo_matrix(A_l_sparse)

    sparse_matrix = {}
    for i, j, v in zip(A_l_coo.row, A_l_coo.col, A_l_coo.data):
        sparse_matrix[(i,j)] = v

    return sparse_matrix


def add_periodicity_edges(G, filename, vertices_to_edges, edges_to_vertices, sparse_matrix, draw= False):
    """
    1. Check if every nonzero element in the adjacency matrix of G is also in the sparse matrix file.
    2. If all edges in G exist in the file, add missing edges from the adjacency matrix file to G.
    """

    # Get adjacency matrix from NetworkX graph
    adj_matrix = nx.to_numpy_array(G)

    # Track missing edges in the file
    missing_edges = []

    # Step 1: Check if every edge in G exists in the sparse matrix file
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:  # Edge exists in G
                if (i, j) not in sparse_matrix and (j, i) not in sparse_matrix:
                    missing_edges.append((i, j))

    # If any edge in G is missing from the file, print an error and stop execution
    if missing_edges:
        # This should be Raised as an error not only printed.
        print("Error: Not all edges in the graph are present in the adjacency matrix file.")
        for edge in missing_edges:
            print(f"Missing edge: {edge}")
        return  # Exit the function if errors are found

    print("All edges in the graph are correctly represented in the txt file.")

    # Step 2: Add edges from the sparse matrix file to G if they are not present in G
    edges_added = []
    edge_count = len(G.edges())
    for (i, j) in sparse_matrix.keys():
        if not G.has_edge(i, j):  # If the edge is not in G, add it
            G.add_edge(i, j, with_labels=True)
            edges_added.append((i, j))
            vertices_to_edges[tuple(sorted((i, j)))] = edge_count
            edges_to_vertices[edge_count] = tuple(sorted((i, j)))
            edge_count += 1

    # # Print added edges
    if edges_added:
        print(f"Added {len(edges_added)} edges to the graph:")
        for edge in edges_added:
            print(f"Added edge: {edge}")
    else:
        print("No additional edges were added to the graph.")
    
    if draw:
        nx.draw(
            G,
            pos=pos_dict,
            node_size=50,  # Adjust node size
            node_color="lightblue",
            with_labels=True,
            font_size=12,  # Increase font size for node labels
            font_color="black"
        )
        nx.draw_networkx_edge_labels(
            G,
            pos=pos_dict,
            edge_labels=vertices_to_edges,
            font_size=10,  # Increase font size for edge labels
            label_pos=0.5,  # Adjust edge label position (closer to the center of edges)
        )
    return G

def generate_phase_flip_circuit(graph, error_prob, vertices_to_edges):
    affected_edges = []
    num_vertices = len(graph.nodes())
    num_edges = len(graph.edges())

    vertices_qubits = QuantumRegister(num_vertices, 'vertices_qubits')
    edges_qubits = QuantumRegister(num_edges, 'edges_qubits')
    # print(num_edges)
    # print(num_vertices)
    cr = ClassicalRegister(num_vertices, 'cr')
    # We will still add classical registers for plaquettes
    qc = QuantumCircuit()
    qc.add_register(vertices_qubits)
    qc.add_register(edges_qubits)
    qc.add_register(cr)

    for j in range(len(edges_qubits)):
        random_number = np.random.rand()
        if random_number < error_prob:
            qc.x(edges_qubits[j])
            affected_edges.append(j)

    # for j in error_prob:
    #     qc.x(edges_qubits[j])
    #     affected_edges.append(j)
    # print('affected_edges', affected_edges)

    for v in range(len(graph.nodes())):
        incident_edges = get_edge_labels_for_vertex(graph, v, vertices_to_edges)
        for edge in incident_edges:
            # qc.h(edges_qubits[edge])
            # qc.cx(edges_qubits[edge], vertices_qubits[v])
            # qc.h(edges_qubits[edge])
            # edges_qubits[edge]
            qc.cx(edges_qubits[edge], vertices_qubits[v])

        qc.barrier()

    for v in range(len(graph.nodes())):
        qc.measure(vertices_qubits[v], cr[v])

    # qubits_of_interest = [edges_qubits[5], vertices_qubits[2], vertices_qubits[3], edges_qubits[149], vertices_qubits[130], vertices_qubits[131]]

    # filtered_circuit = QuantumCircuit(*qubits_of_interest, cr[0], cr[1])

    # # Copy only the instructions that involve the selected qubits
    # for instruction in qc.data:
    #     qubits, clbits = instruction[1], instruction[2]
    #     if any(q in qubits_of_interest for q in qubits):
    #         filtered_circuit.append(instruction[0], qubits, clbits)
    #
    # # Draw the filtered circuit
    # filtered_circuit.draw("mpl")

    return qc, affected_edges
def generate_bit_flip_circuit(graph, error_prob, vertices_to_edges):
    affected_edges = []
    num_vertices = len(graph.nodes())
    num_edges = len(graph.edges())

    vertices_qubits = QuantumRegister(num_vertices, 'vertices_qubits')
    edges_qubits = QuantumRegister(num_edges, 'edges_qubits')
    # print(num_edges)
    # print(num_vertices)
    cr = ClassicalRegister(num_vertices, 'cr')
    # We will still add classical registers for plaquettes
    qc = QuantumCircuit()
    qc.add_register(vertices_qubits)
    qc.add_register(edges_qubits)
    qc.add_register(cr)

    for j in range(len(edges_qubits)):
        random_number = np.random.rand()
        if random_number < error_prob:
            qc.x(edges_qubits[j])
            affected_edges.append(j)

    # for j in error_prob:
    #     qc.x(edges_qubits[j])
    #     affected_edges.append(j)
    # print('affected_edges', affected_edges)

    for v in range(len(graph.nodes())):
        incident_edges = get_edge_labels_for_vertex(graph, v, vertices_to_edges)
        for edge in incident_edges:
            # qc.h(edges_qubits[edge])
            # qc.cx(edges_qubits[edge], vertices_qubits[v])
            # qc.h(edges_qubits[edge])
            # edges_qubits[edge]
            qc.cx(edges_qubits[edge], vertices_qubits[v])

        qc.barrier()

    for v in range(len(graph.nodes())):
        qc.measure(vertices_qubits[v], cr[v])

    # qubits_of_interest = [edges_qubits[5], vertices_qubits[2], vertices_qubits[3], edges_qubits[149], vertices_qubits[130], vertices_qubits[131]]

    # filtered_circuit = QuantumCircuit(*qubits_of_interest, cr[0], cr[1])

    # # Copy only the instructions that involve the selected qubits
    # for instruction in qc.data:
    #     qubits, clbits = instruction[1], instruction[2]
    #     if any(q in qubits_of_interest for q in qubits):
    #         filtered_circuit.append(instruction[0], qubits, clbits)
    #
    # # Draw the filtered circuit
    # filtered_circuit.draw("mpl")

    return qc, affected_edges


def generate_syndrome_graph(indices, graph):
    syndrome_graph = nx.Graph()
    for v in indices:
        syndrome_graph.add_node(v, label = True)

    for i, ver1 in enumerate(indices):
        for j, ver2 in enumerate(indices[i+1:], start=i+1):
            weight = nx.shortest_path_length(graph, source=ver1, target=ver2)
            syndrome_graph.add_edge(ver1, ver2, weight=weight)

    # Draw the syndrome graph
    pos = nx.spring_layout(syndrome_graph)  # Positioning for better visualization
    # nx.draw(syndrome_graph, pos, with_labels=True, node_color="lightblue", node_size=500)

    # Add edge labels showing the weights
    edge_labels_with_weight = nx.get_edge_attributes(syndrome_graph, 'weight')  # Retrieve weights
    # nx.draw_networkx_edge_labels(syndrome_graph, pos, edge_labels=edge_labels_with_weight)

    # Show the graph
    # plt.show()
    return syndrome_graph


def find_correction_paths(syndrome_graph, circuit_graph):
    matching = nx.algorithms.matching.min_weight_matching(syndrome_graph)

    correction_paths = []
    # Print the matching
    # print("Minimum Weight Perfect Matching:", matching)
    for edge in matching:
        # print(edge, "with weight", syndrome_graph.edges[edge]['weight'])
        path = nx.shortest_path(circuit_graph, source=edge[0], target=edge[1])
        # print('path', path)
        correction_paths.append(path)
    return correction_paths


def get_edge_from_v1_v2(v1, v2, vertices_to_edges):
    tup = tuple(sorted((v1, v2)))
    return vertices_to_edges[tup]


# This function needs to be corrected, v is unused and p is not defined.
def find_all_plaquette_edges(original_graph, vertices_to_edges):
    p = 8
    plaquettes = nx.cycle_basis(original_graph)
    
    all_plaquette_edges = []
    for plaquette_vertices in plaquettes:
        plaquette_edges = []
        i = 0
        for v in plaquette_vertices:
            # print(plaquette_vertices)
            plaquette_edges.append(
                get_edge_from_v1_v2(plaquette_vertices[i], plaquette_vertices[(i + 1) % p], vertices_to_edges))
            i = i + 1
        # all_plaquette_edges.append((plaquette_vertices, plaquette_edges))
    return all_plaquette_edges



# if the syndrome measurement plus the affected errors span two neighboring, then the function will detect a logic error while there is none
# print('affected_edges', sorted(affected_edges))
def get_logical_error(affected_edges, correction_paths, all_plaquette_edges, vertices_to_edges):
    # This p gave me an error.
    p = 8
    affected_edges_set = set(affected_edges)
    all_correction_edges = set()
    for correction_path in correction_paths:
        for i in range(len(correction_path) - 1):
            all_correction_edges.add(get_edge_from_v1_v2(correction_path[i], correction_path[i + 1], vertices_to_edges))
    # print('correction_edg', sorted(all_correction_edges))

    if all_correction_edges == affected_edges_set:
        return False

    # remove the common edges between affected and correction edges
    intersection = all_correction_edges.intersection(affected_edges_set)
    union = all_correction_edges.union(affected_edges_set)
    potential_plaquettes = union - intersection

    # print('potential_plaquettes', potential_plaquettes)
    # if something remains, check if they form plaquettes by looping over the 17 possible plaquettes
    if len(potential_plaquettes) % p != 0:
        return True

    for _, v in all_plaquette_edges:
        if set(v).issubset(potential_plaquettes):
            potential_plaquettes -= set(v)
    if len(potential_plaquettes) > 0:
        return True
    return False


def run_trial(args):
    ep, graph, original_graph, vertices_to_edges, all_plaquette_edges = args
    qcc, affected_edges = generate_bit_flip_circuit(graph, ep, vertices_to_edges)
    simulator = AerSimulator()
    result = simulator.run(qcc, shots=1).result().get_counts()
    syndrome_measurement = next(iter(result))[::-1]
    indices = [j for j, bit in enumerate(syndrome_measurement) if bit == "1"]
    syndrome_graph = generate_syndrome_graph(indices, graph)
    correction_paths = find_correction_paths(syndrome_graph, graph)
    is_err = get_logical_error(affected_edges, correction_paths, all_plaquette_edges, vertices_to_edges)
    return is_err


def error_graph(graph, original_graph, vertices_to_edges, error_probabilities):
    trials = 5000
    error_percentages = []
    all_plaquette_edges = find_all_plaquette_edges(original_graph, vertices_to_edges)
    for ep in error_probabilities:
        print(f"Processing error probability {ep} for the graph with {graph.number_of_edges()} qubits")
        # Prepare the arguments for each trial
        args_list = [(ep, graph, original_graph, vertices_to_edges, all_plaquette_edges) for _ in range(trials)]
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(run_trial, args_list)
        count = sum(results)
        error_percentages.append(count / trials)
    return error_percentages


if __name__ == '__main__':
    p = 8
    q = 3
    p_B = 8
    q_B = 8

    CT_matrices_dic = {
        9: [[2, 6, 1, 5, 8, 9, 3, 7, 4],
            [3, 1, 7, 9, 4, 2, 8, 5, 6],
            [4, 5, 9, 3, 1, 8, 6, 2, 7],
            [5, 8, 4, 1, 2, 7, 9, 6, 3],
            [6, 9, 2, 8, 7, 4, 1, 3, 5],
            [7, 3, 8, 6, 9, 1, 5, 4, 2],
            [8, 7, 5, 2, 6, 3, 4, 9, 1],
            [9, 4, 6, 7, 3, 5, 2, 1, 8]],

        12: [[2, 9, 1, 10, 6, 8, 4, 3, 7, 12, 5, 11],
             [3, 1, 8, 7, 11, 5, 9, 6, 2, 4, 12, 10],
             [4, 10, 7, 5, 1, 2, 11, 9, 12, 6, 3, 8],
             [5, 6, 11, 1, 4, 10, 3, 12, 8, 2, 7, 9],
             [6, 8, 5, 2, 10, 12, 1, 11, 3, 9, 4, 7],
             [7, 4, 9, 11, 3, 1, 12, 2, 10, 5, 8, 6],
             [8, 3, 6, 9, 12, 11, 2, 5, 1, 7, 10, 4],
             [9, 7, 2, 12, 8, 3, 10, 1, 4, 11, 6, 5]],

        15: [ [ 2, 7, 1, 10, 8, 3, 9, 6, 4, 13, 5, 15, 14, 12, 11 ],
              [ 3, 1, 6, 9, 11, 8, 2, 5, 7, 4, 15, 14, 10, 13, 12 ],
              [ 4, 10, 9, 12, 1, 7, 13, 2, 14, 15, 3, 8, 11, 5, 6 ],
              [ 5, 8, 11, 1, 14, 15, 6, 12, 3, 2, 13, 4, 7, 9, 10 ],
              [ 6, 3, 8, 7, 15, 5, 1, 11, 2, 9, 12, 13, 4, 10, 14 ],
              [ 7, 9, 2, 13, 6, 1, 4, 3, 10, 14, 8, 11, 12, 15, 5 ],
              [ 8, 6, 5, 2, 12, 11, 3, 15, 1, 7, 14, 10, 9, 4, 13 ],
              [ 9, 4, 7, 14, 3, 2, 10, 1, 13, 12, 6, 5, 15, 11, 8 ] ]
                    }

    plt.figure(figsize=(8, 6))  # Set figure size once before the loop

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Define different colors
    error_probabilities = [0.01 * j for j in range(26)]
    for idx, N in enumerate([9, 12, 15]):
        graph_vertices = generate_vertices(p, q, p_B, q_B, N)
        sparse_matrix = create_adjacency_matrix(N, CT_matrices_dic[N], p_B)
        N_graph, graph_vertices_to_edges, graph_edges_to_vertices = generate_hyperbolic_graph(graph_vertices)
        N_graph_original = copy.deepcopy(N_graph)

        # Example usage
        filename = f"sparse_matrix_{N}.txt"  # Replace with the actual file path
        periodic_N9_graph = add_periodicity_edges(N_graph, filename, graph_vertices_to_edges, graph_edges_to_vertices, sparse_matrix)

        N_err_percentage = error_graph(periodic_N9_graph, N_graph_original, graph_vertices_to_edges, error_probabilities)
        N_num_qubits = periodic_N9_graph.number_of_edges()

        
        k = 2*N+1
        plt.plot(error_probabilities, N_err_percentage, marker='o', linestyle='-',
                 color=colors[idx % len(colors)], label=f'[[n={N_num_qubits},k={k}]]')

    # Add labels and a title
    plt.xlabel('Error Probabilities')
    plt.ylabel('Error Percentages')
    plt.title('Error Threshold Graph')
    plt.legend()
    plt.grid(True)

    plt.show()  # Show all curves in one figure

    end = time.time()
    print(end - start)