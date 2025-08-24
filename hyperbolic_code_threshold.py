# Copyright 2025 Ahmed Adel Mahmoud and Kamal Mohamed Ali

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import multiprocessing
import time
from collections import Counter, defaultdict
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from matplotlib import colormaps
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import norm
simulator = AerSimulator()

start = time.time()


def hyperbolic_distance(z1: complex, z2: complex):
    """Calculate the hyperbolic distance between two points in the complex plane."""

    d = np.arccosh(1 + (2 * (abs(z1 - z2)) ** 2 / ((1 - abs(z1) ** 2) * (1 - abs(z2) ** 2))))
    return d


def euclidean_distance(z1: complex, z2: complex):
    d = abs(z1 - z2)
    return d


def unit_cell_positions(p: int, q: int):
    """Find the positions of the vertices in the unit cell of the hyperbolic lattice {p,q}"""

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


def rotation_matrix(phi: complex):
    """ This function generates the rotation matrices used to generate the Fuchsian group generators; see eq (20) in the paper"""
    return np.array([[np.exp(1j * phi / 2), 0], [0, np.exp(-1j * phi / 2)]])


def fuchsian_generators(p_B: int, q_B: int):
    """ This function generates the generators of the Fuchsian group as well as their inverses and returns them in a list."""
    alpha = 2 * np.pi / p_B
    beta = 2 * np.pi / q_B
    sigma = np.sqrt((np.cos(alpha) + np.cos(beta)) / (1 + np.cos(beta)))
    gamma1 = 1 / (np.sqrt(1-sigma**2)) * np.array([[1, sigma], [sigma, 1]])
    FG_generators = []
    for mu in range(0, int(p_B / 2)):
        gamma_j = rotation_matrix(mu * alpha) @ gamma1 @ rotation_matrix(-mu * alpha)
        FG_generators.append(gamma_j)
        FG_generators.append(np.linalg.inv(gamma_j))
    return FG_generators


def create_new_vertex(vertex_position: complex, translation_matrix: np.array):
    """ This function generates the position of the vertex produced by applying an element of the Fuchsian group to a vertex in the unit cell"""
    v = translation_matrix @ np.array([vertex_position, 1])
    new_vertex_position = v[0] / v[1]
    return new_vertex_position


def generate_vertices(p: int, q: int, p_B: int, q_B: int, N_B: int):
    # Create the unit cell of the {p,q} lattice
    unit_cell = unit_cell_positions(p, q)
    # Generate the Fuchsian generators of the Bravais lattice.
    group_generators = fuchsian_generators(p_B, q_B)
    # This list is used to store vertices that are not in the unit cell.
    outer_rings = []

    # If N_B= 1, we return the unit cell.
    if N_B == 1:
        return unit_cell

    # We start by producing p_B new faces in the Bravais lattice by applying all the Fuchsian group generators
    # to the unit cell.
    for generator in group_generators:
        for vertex in unit_cell:
            new_vertex = create_new_vertex(vertex, generator)
            outer_rings.append(new_vertex)

    # Next, we create more faces by applying more elements of the Fuchsian group to the unit cell.
    # The number of new faces is the length of the extra_generators_indices list.
    extra_generators_indices = []
    if N_B == 12:
        extra_generators_indices = [(1, 2), (0, 3), (2, 2)]
    elif N_B == 16:
        extra_generators_indices = [[0, 0], [0, 6], [0, 7], [1, 6], [1, 7], [2, 7], [3, 6]]

    elif N_B == 20:
        extra_generators_indices = [[0, 2], [0, 3], [0, 5], [1, 2], [1, 3], [1, 4], [2, 4], [2, 6], [2, 7], [3, 6],
                                    [3, 7]]

    elif N_B == 25:
        extra_generators_indices = [[0, 0], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 1],
                                    [1, 2], [1, 5], [1, 6], [1, 7], [2, 4], [2, 6], [3, 5], [3, 7]]
    elif N_B == 35:
        # NSG[61332]
        extra_generators_indices = [[0, 0], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 1], [1, 2], [1, 3],
                                    [1, 4],
                                    [1, 5], [1, 6], [1, 7], [2, 2], [2, 4], [2, 6], [3, 3], [3, 5], [3, 7], [4, 4],
                                    [4, 7], [5, 5], [5, 6], [6, 6], [7, 7]]
    elif N_B == 48:
        extra_generators_indices = [ [ 0, 3 ], [ 0, 4 ], [ 0, 5 ], [ 6, 0 ], [ 4, 6 ], [ 1, 2 ], [ 1, 4 ], [ 1, 5 ], [ 5, 7 ], [ 7, 1 ], [ 2, 2 ],
                                      [ 2, 4 ], [ 2, 5 ], [ 2, 6 ], [ 2, 7 ], [ 3, 4 ], [ 3, 5 ], [ 3, 6 ], [ 3, 7 ], [ 4, 4 ], [ 0, 7 ], [ 7, 4 ],
                                      [ 5, 5 ], [ 6, 5 ], [ 1, 6 ], [ 0, 6 ], [ 5, 6 ], [ 6, 6 ], [ 1, 7 ], [ 4, 7 ], [ 7, 7 ], [0,4,4], [0,4,7], [0,5,6], [0,5,7], [6,0,5], [4,6,1], [4,6,4],[2,2,4]  ]

    for index_pair in extra_generators_indices:
        if len(index_pair) == 2:
            # For each pair of indices, we generate an element of the Fuchsian group by matrix multiplication.
            fuchsian_group_element = group_generators[index_pair[0]] @ group_generators[index_pair[1]]
        elif len(index_pair) == 3:
            fuchsian_group_element = group_generators[index_pair[2]] @ group_generators[index_pair[0]] @ group_generators[index_pair[1]]
        else:
            raise ValueError("index pair has length > 3")

        for vertex in unit_cell:
            new_vertex = create_new_vertex(vertex, fuchsian_group_element)
            outer_rings.append(new_vertex)

    final_vertices = unit_cell + outer_rings

    return final_vertices


def generate_hyperbolic_graph(vertices: list, draw=False):
    """Generate a Networkx graph from the given vertices positions.

    Args:
        vertices: list of positions for vertices
        draw: whether to plot the graph

    Returns:
        G: a NetworkX graph constructed from the vertices. This graph represents the hyperbolic lattice
        adj_G: the adjacency matrix of G
        vertices_to_edges: a dict mapping vertices to the edges connecting them
        edges_to_vertices: a dict mapping edges to the vertices at their endpoints
        pos_dict: a dict mapping each vertex in the graph to its position

    """
    vertices_to_edges = {}

    edges_to_vertices = {}

    d0 = hyperbolic_distance(vertices[0], vertices[1])

    coords = [(v.real, v.imag) for v in vertices]

    G = nx.Graph()

    for idx, pos in enumerate(coords):
        G.add_node(idx, pos=pos, label=True)

    edge_count = 0
    n = len(vertices)
    adj_G = np.zeros((n, n))

    # Algorithm 1 Step 3 (part 2)
    # Add edges to the graph and construct the adjacency matrix.
    for i, pos1 in enumerate(vertices):
        for j, pos2 in enumerate(vertices[i+1:], start=i+1):
            if hyperbolic_distance(pos1, pos2) < (d0 + 0.1):
                adj_G[i][j] = 1
                adj_G[j][i] = 1
                G.add_edge(i, j, with_labels=True)
                vertices_to_edges[(i, j)] = edge_count
                edges_to_vertices[edge_count] = (i, j)
                edge_count += 1

    pos_dict = {idx: (pos.real, pos.imag) for idx, pos in enumerate(vertices)}

    if draw:
        plt.figure(figsize=(25, 25))
        nx.draw(
            G,
            pos=pos_dict,
            node_size=5,
            node_color="lightblue",
            with_labels=True,
            font_size=5,
            font_color="black"
        )

        nx.draw_networkx_edge_labels(
            G,
            pos=pos_dict,
            edge_labels=vertices_to_edges,
            font_size=5,
            label_pos=0.5,
        )

        plt.axis("equal")
        plt.show()

    return G, adj_G, vertices_to_edges, edges_to_vertices, pos_dict


def get_edge_labels_for_vertex(G: nx.Graph, vertex: int, vertices_to_edges: dict):
    """Takes as input a graph and a vertex label and returns the labels of all edges incident on this vertex."""
    incident_edges = list(G.edges(vertex))
    incident_edge_labels = {edge: vertices_to_edges[tuple(sorted(edge))] for edge in incident_edges}
    return incident_edge_labels.values()


def get_edge_from_v1_v2(v1: int, v2: int, vertices_to_edges: dict):
    """Takes labels for two vertices for an edge and returns the corresponding edge label.
    Edge labels are easier to work and debug with."""
    tup = tuple(sorted((v1, v2)))
    return vertices_to_edges[tup]


def create_adjacency_matrix(N_B: int, CT: list, p_B: int, p: int,
                            q: int):
    # Get a list of positions of vertices in the unit cell.
    unit_cell = unit_cell_positions(p, q)
    n = len(unit_cell)

    unit_cell_graph = generate_hyperbolic_graph(unit_cell)
    # Adjacency matrix of the unit cell
    V = nx.to_numpy_array(unit_cell_graph[0])
    I = np.identity(N_B)
    A_l = np.kron(I, V)

    # Initialize the inter-cell matrices. These matrices dictate how to glue different unit cells together in the graph.
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

    # Perform the update to A_l based on CT
    for alpha in range(p_B):  # Iterate over the T_matrices
        for i in range(N_B):
            # Create U matrix
            U = np.zeros((N_B, N_B))
            j = CT[alpha][i] - 1  # Adjust indexing for Python (0-based)
            U[i, j] = 1
            # Update A_l
            A_l += np.kron(U, T_matrices[alpha])

    # Convert A_l to a sparse matrix. Do we have to do this step or can we just convert it into COO right away.?
    A_l_sparse = csr_matrix(A_l)

    # Convert to COO format to access row, col, and data attributes
    A_l_coo = coo_matrix(A_l_sparse)

    sparse_matrix = {(i, j): v for i, j, v in zip(A_l_coo.row, A_l_coo.col, A_l_coo.data)}

    for (i, j), v in sparse_matrix.items():
        if (j, i) not in sparse_matrix:
            raise ValueError("The sparse matrix is not symmetric")

    # Step 2: Count occurrences of each vertex
    vertex_count = Counter()
    for u, v in sparse_matrix.keys():
        vertex_count[u] += 1
        vertex_count[v] += 1

    # Step 3: Find redundant nodes
    degree_check = all(count == 2 * q for node, count in vertex_count.items())
    if not degree_check:
        print([(node, count) for node, count in vertex_count.items() if count != 2 * q])
        raise ValueError("Some nodes do not appear 8 times in the sparse matrix")

    return sparse_matrix


def add_periodicity_edges(original_graph: nx.Graph, vertices_to_edges: dict, edges_to_vertices: dict,
                          sparse_matrix: dict, vertices_positions: dict, draw=False):
    """
    Finds all the new edges based on periodic boundary condition, then adds those edges to a copy of the original graph.
    1. Check if every nonzero element in the adjacency matrix of G is also in the sparse matrix.
    2. If all edges in G exist in the matrix, add missing edges from the adjacency matrix to G.
    """

    periodic_G = copy.deepcopy(original_graph)

    # Add edges from the sparse matrix to G if they are not present in G
    edges_added = []
    edge_count = len(periodic_G.edges())
    for (i, j) in sparse_matrix.keys():
        if not periodic_G.has_edge(i, j):
            if i in periodic_G.nodes and j in periodic_G.nodes:
                periodic_G.add_edge(int(i), int(j), with_labels=True)
                vertices_to_edges[(i, j)] = edge_count
                edges_to_vertices[edge_count] = (i, j)
                edges_added.append((i, j))
                edge_count += 1
            else:
                raise Exception(f"Could not add edge; nodes ({i}, {j}) not in graph")

    if draw:
        plt.figure(figsize=(20, 20))
        nx.draw(
            periodic_G,
            pos=vertices_positions,
            node_size=20,
            node_color="lightblue",
            with_labels=True,
            font_size=6,
            font_color="black"
        )
        nx.draw_networkx_edge_labels(
            periodic_G,
            pos=vertices_positions,
            edge_labels=vertices_to_edges,
            font_size=10,
            label_pos=0.5,
        )
    return periodic_G


def surface_genus(periodic_graph: nx.Graph, p: int):
    F = 2 * periodic_graph.number_of_edges() / p
    E = periodic_graph.number_of_edges()
    V = periodic_graph.number_of_nodes()
    Euler_characteristic = F - E + V
    genus = int((2 - Euler_characteristic) / 2)
    return genus


def generate_bit_flip_circuit(graph: nx.Graph, error_prob: float, vertices_to_edges: dict):

    """Takes the graph as input and generates the corresponding quantum circuit for bit flip error in Qiskit. 
    The bit flip circuit applies a CX between each edge qubit and its incident vertices. 
    A random X error is introduced randomly based on input error probability."""

    affected_edges = []
    num_vertices = len(graph.nodes())
    num_edges = len(list(graph.edges()))

    vertices_qubits = QuantumRegister(num_vertices, 'vertices_qubits')
    edges_qubits = QuantumRegister(num_edges, 'edges_qubits')
    cr = ClassicalRegister(num_vertices, 'cr')
    qc = QuantumCircuit()
    qc.add_register(vertices_qubits)
    qc.add_register(edges_qubits)
    qc.add_register(cr)

    for j in range(len(edges_qubits)):
        random_number = np.random.rand()
        if random_number < error_prob:
            qc.x(edges_qubits[j])
            affected_edges.append(j)

    for v in range(len(graph.nodes())):
        incident_edges = get_edge_labels_for_vertex(graph, v, vertices_to_edges)
        for edge in incident_edges:
            qc.cx(edges_qubits[edge], vertices_qubits[v])

    for v in range(len(graph.nodes())):
        qc.measure(vertices_qubits[v], cr[v])

    return qc, affected_edges


def generate_phase_flip_circuit(periodic_G: nx.Graph,  err_prob: float, graph_vertices_to_edges: dict, all_faces: list):

    affected_edges = []
    num_plaquettes = len(all_faces)
    num_edges = periodic_G.number_of_edges()

    plaquette_ancilla_qubits = QuantumRegister(num_plaquettes, 'plaquette_ancilla_qubits')
    edges_qubits = QuantumRegister(num_edges, 'edges_qubits')
    cr = ClassicalRegister(num_plaquettes, 'cr')

    qc = QuantumCircuit()
    qc.add_register(plaquette_ancilla_qubits)
    qc.add_register(edges_qubits)
    qc.add_register(cr)

    # H on all qubits to change basis
    for i in range(num_plaquettes):
        qc.h(plaquette_ancilla_qubits[i])
    for j in range(num_edges):
        qc.h(edges_qubits[j])

    # add random Z errors
    for j in range(num_edges):
        random_number = np.random.rand()
        if random_number < err_prob:
            qc.z(edges_qubits[j])
            affected_edges.append(j)
    qc.barrier()

    # CX between centre ancilla and neighboring edges
    for i, plaquette in enumerate(all_faces):
        for edge in plaquette:
            edge_label = get_edge_from_v1_v2(edge[0], edge[1], graph_vertices_to_edges)
            qc.cx(plaquette_ancilla_qubits[i], edges_qubits[edge_label])
    qc.barrier()

    # H on all again
    for i in range(num_plaquettes):
        qc.h(plaquette_ancilla_qubits[i])
    for j in range(num_edges):
        qc.h(edges_qubits[j])
    qc.barrier()

    # Z measurement
    for v in range(num_plaquettes):
        qc.measure(plaquette_ancilla_qubits[v], cr[v])

    return qc, affected_edges


def generate_n_fold_trivial_cycles(all_faces:list, max_n:int):

    def symmetric_diff(c1, c2):
        return frozenset(c1.symmetric_difference(c2))

    # Step 1: Compute two-fold intersections
    two_fold_cycles = set()
    intersection_edges = {}
    for i, f1 in enumerate(all_faces):
        for j, f2 in enumerate(all_faces[i+1:], start=i+1):
            inter = f1.intersection(f2)
            if len(inter) == 1:
                cycle = symmetric_diff(f1, f2)
                two_fold_cycles.add((frozenset({i, j}), cycle))
                intersection_edges[(i, j)] = next(iter(inter))

    all_trivial_cycles = {2: two_fold_cycles}

    # Step 2: Build higher order cycles
    for n in range(3, max_n + 1):
        prev_cycles = all_trivial_cycles[n - 1]
        current_cycles = set()
        for indices, c in prev_cycles:
            for k, f in enumerate(all_faces):
                if k not in indices:
                    overlap = len(c.intersection(f))
                    if 1 <= overlap <= n - 1:
                        new_cycle = symmetric_diff(c, f)
                        new_indices = indices | {k}
                        current_cycles.add((new_indices, new_cycle))
        all_trivial_cycles[n] = current_cycles
        # length_counts = Counter(len(cycle) for _, cycle in current_cycles)
        # breakdown = ", ".join(f"{count} of length {length}" for length, count in sorted(length_counts.items()))
        # print(f"{n}-fold cycles: {len(current_cycles)} total ({breakdown})")

    # Final flattening
    trivial_cycles_set = {cycle for cycle_set in all_trivial_cycles.values() for _, cycle in cycle_set}
    return trivial_cycles_set, intersection_edges


def HCB_Algorithm(original_graph: nx.Graph, periodic_graph: nx.Graph, genus :int, p: int, dual=False):
    cycle_basis = []
    all_faces = []
    X_logicals = []
    potential_logicals_dict = defaultdict(list)

    num_plaquettes = int(2 * periodic_graph.number_of_edges() / p)
    # print(f"num_of_plaquettes should be {num_plaquettes}")

    if dual:
        max_n = 6
        l_threshold = 3*p-1
    else:
        max_n = 3
        l_threshold = 2 * p + 4

    intital_p_cycles = [cycle_to_edges(cycle) for cycle in list(nx.simple_cycles(original_graph, p))]

    # Append the cycle to the list of cycle basis and the list of all faces
    for cycle in intital_p_cycles:
        if len(cycle) == p:
            cycle_basis.append(cycle)
            all_faces.append(cycle)

    # Find F-1 plaquettes and add them to cycle basis as well as all faces
    for cycle in list(nx.simple_cycles(periodic_graph, l_threshold)):
        cycle_edges = cycle_to_edges(cycle)
        if len(cycle) == p:
            valid_plaquette = all(len(cycle_edges.intersection(plaquette)) <= 1 for plaquette in all_faces)
            if valid_plaquette and len(cycle_basis) < num_plaquettes-1:
                cycle_basis.append(cycle_edges)
                all_faces.append(cycle_edges)
            elif valid_plaquette and len(cycle_basis) == num_plaquettes-1:
                all_faces.append(cycle_edges)
        elif len(cycle) > p:
            potential_logicals_dict[len(cycle)].append(cycle_edges)

    if len(all_faces) > num_plaquettes:
        raise ValueError(f"len(all_faces) = {len(all_faces)}, while num_plaquettes = {num_plaquettes}")

    # for length, cycles in potential_logicals_dict.items():
        # print(f"The number logicals of length {length} is {len(cycles)}")

    # Generate trivial cycles obtained as products of plaquettes.
    trivial_p_cycles, intersection_edges = generate_n_fold_trivial_cycles(all_faces=all_faces, max_n=max_n)

    # Flatten dictionary into a single list
    potential_logicals = []
    for length in sorted(potential_logicals_dict):
        potential_logicals.extend(potential_logicals_dict[length])

    # print(f"length of potential_logicals before dedupliation is {len(potential_logicals)}")

    potential_logicals_non_trivial = [
        cycle for cycle in potential_logicals if cycle not in trivial_p_cycles
    ]

    # print(f"length of potential_logicals after dedupliation is {len(potential_logicals_non_trivial)}")
    if dual:
        # print("The length of potential_Z_logicals is ", len(potential_logicals))
        return cycle_basis, intital_p_cycles, all_faces, intersection_edges, potential_logicals

    while len(X_logicals) < 2 * genus:
        found = False
        for logical_edges in potential_logicals_non_trivial:
            commutation_check = all(
                len(logical_edges.intersection(X_logical)) % 2 == 0 for X_logical in X_logicals)
            if commutation_check:
                X_logicals.append(logical_edges)
                cycle_basis.append(logical_edges)
                potential_logicals_non_trivial.remove(logical_edges)
                found = True
                break
        if not found:
            raise ValueError("Could not find a valid X_logical, len(X_logicals) = {}".format(len(X_logicals)))

    return cycle_basis, intital_p_cycles, all_faces, intersection_edges, X_logicals


def find_Z_logicals(X_logicals: list, potential_Z_logicals: list, intersection_edges: dict):
    k = 0
    Z_logicals = []
    while len(Z_logicals) < len(X_logicals):
        found = False
        for cycle in potential_Z_logicals:
            logical_edges = dual_logical_to_Z_l(dual_X_logical=cycle, intersection_edges=intersection_edges)
            logical_vertices = set()
            duplicated_vertex = False
            for (u,v) in logical_edges:
                if u not in logical_vertices and v not in logical_vertices:
                    logical_vertices.add(u)
                    logical_vertices.add(v)
                else:
                    duplicated_vertex = True
            if duplicated_vertex:
                continue
            commutation_check = all(
                len(logical_edges.intersection(Z_logical)) % 2 == 0 for Z_logical in Z_logicals)
            anti_commutation_check = len(logical_edges.intersection(X_logicals[k])) % 2 == 1

            if commutation_check and anti_commutation_check:
                Z_logicals.append(logical_edges)
                potential_Z_logicals.remove(cycle)
                found = True
                k += 1
                break
        if not found:
            raise ValueError("Could not find a valid Z_logical, len(Z_logicals) = {}".format(len(Z_logicals)))
    return Z_logicals


def cycle_to_edges(cycle: list):
    return frozenset(tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)]))) for i in range(len(cycle)))


def dual_logical_to_Z_l(dual_X_logical: set, intersection_edges: dict):
    Z_l = set()
    for pair in dual_X_logical:
        intersection_e = intersection_edges[pair]
        Z_l.add(intersection_e)
    return Z_l


def generate_dual_graph(faces: list, intersection_edges: dict, p: int, vertices_to_edges: dict, G_pos_dict: dict, draw=False):
    """Generate the dual graph which replaces each plaquette with a vertix.
    Used for computing the phase flip circuit."""
    G_dual_graph = nx.Graph()

    pos_dict = {}

    # For each face in graph, add a node to dual graph with its position at the center of all the face's nodes
    for i, plaquette in enumerate(faces):
        plaquette_vertices = set()
        for pair in plaquette:
            plaquette_vertices.add(pair[0])
            plaquette_vertices.add(pair[1])

        x_pos = 0.0
        y_pos = 0.0
        for v in plaquette_vertices:
            x_pos += G_pos_dict[v][0]
            y_pos += G_pos_dict[v][1]

        pos = (x_pos / p, y_pos / p)
        pos_dict[i] = pos

        G_dual_graph.add_node(i, pos=pos, label=True)

    for (i, j), edge in intersection_edges.items():
        if i in pos_dict.keys() and j in pos_dict.keys():
            label = get_edge_from_v1_v2(edge[0], edge[1], vertices_to_edges)
            G_dual_graph.add_edge(i, j, label=label)

    if draw:
        plt.figure(figsize=(10, 10))

        nx.draw(
            G_dual_graph,
            pos=pos_dict,
            node_size=20,
            node_color="lightblue",
            with_labels=True,
            font_size=12,
            font_color="black"
        )
        edge_labels = nx.get_edge_attributes(G_dual_graph, "label")

        nx.draw_networkx_edge_labels(
            G_dual_graph,
            pos=pos_dict,
            edge_labels=edge_labels,
            font_size=10,
            label_pos=0.5,
        )
        plt.show()

    return G_dual_graph


def generate_syndrome_graph(indices: list, graph: nx.Graph, draw=False):
    syndrome_graph = nx.Graph()
    for v in indices:
        syndrome_graph.add_node(v, label=True)

    for i, ver1 in enumerate(indices):
        for j, ver2 in enumerate(indices[i + 1:], start=i + 1):
            weight = nx.shortest_path_length(graph, source=ver1, target=ver2)
            syndrome_graph.add_edge(ver1, ver2, weight=weight)

    # Draw the syndrome graph
    if draw:
        pos = nx.spring_layout(syndrome_graph)  # Positioning for better visualization
        nx.draw(syndrome_graph, pos, with_labels=True, node_color="lightblue", node_size=500)

        edge_labels_with_weight = nx.get_edge_attributes(syndrome_graph, 'weight')
        nx.draw_networkx_edge_labels(syndrome_graph, pos, edge_labels=edge_labels_with_weight)

        # Show the graph
        plt.show()
    return syndrome_graph


def find_correction_paths(syndrome_graph: nx.Graph, periodic_graph: nx.Graph):
    # Find the MWPM from the syndrome graph because edges are weights
    matching = nx.algorithms.matching.min_weight_matching(syndrome_graph)

    correction_paths = []
    for edge in matching:
        path = nx.shortest_path(periodic_graph, source=edge[0], target=edge[1])
        correction_paths.append(path)
    return correction_paths


def logical_operators_to_edges(logical_operators: list, vertices_to_edges: dict):
    """Convert logical operators from a list of pairs of vertices (v1,v2) to edge labels."""
    logical_operators_edges = []

    for logical_operator in logical_operators:
        lg_edges = set()
        for pair in logical_operator:
            lg_edges.add(
                get_edge_from_v1_v2(pair[0], pair[1], vertices_to_edges))
        logical_operators_edges.append(lg_edges)
    return logical_operators_edges


def get_logical_error(affected_edges: list, correction_paths: list, vertices_to_edges: dict, logical_operators: list):
    logical_operators_edges = logical_operators_to_edges(logical_operators, vertices_to_edges)

    affected_edges_set = set(affected_edges)
    all_correction_edges = set()
    for correction_path in correction_paths:
        for i in range(len(correction_path) - 1):
            all_correction_edges.add(get_edge_from_v1_v2(correction_path[i], correction_path[i + 1], vertices_to_edges))

    if all_correction_edges == affected_edges_set:
        return False

    # remove the common edges between affected and correction edges
    potential_logical_error = all_correction_edges.symmetric_difference(affected_edges_set)

    for logical_operator in logical_operators_edges:
        common_edges = potential_logical_error.intersection(logical_operator)
        if len(common_edges) % 2 == 1:
            return True

    return False


def run_trial(args):
    periodic_graph, vertices_to_edges, ep, logical_operators = args
    qcc, affected_edges = generate_bit_flip_circuit(periodic_graph, ep, vertices_to_edges)
    # qcc, affected_edges = generate_phase_flip_circuit(periodic_graph, ep, vertices_to_edges, all_faces)
    result = simulator.run(qcc, shots=1).result().get_counts()
    syndrome_measurement = next(iter(result))[::-1]
    indices = [j for j, bit in enumerate(syndrome_measurement) if bit == "1"]
    syndrome_graph = generate_syndrome_graph(indices, periodic_graph, draw=False)
    correction_paths = find_correction_paths(syndrome_graph, periodic_graph)
    is_err = get_logical_error(affected_edges, correction_paths, vertices_to_edges, logical_operators)
    return is_err


def error_graph(periodic_graph, vertices_to_edges, error_probabilities, logical_operators, trials,
                confidence_level=0.95):

    means = []
    lower_bounds = []
    upper_bounds = []
    z = norm.ppf(1 - (1 - confidence_level) / 2)  # e.g., z=1.96 for 95% confidence

    for ep in error_probabilities:
        # print(f"Processing error probability {ep} for the graph with {periodic_graph.number_of_edges()} qubits")
        args_list = [(periodic_graph, vertices_to_edges, ep, logical_operators) for _ in range(trials)]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(run_trial, args_list)

        results = np.array(results)
        mean_error = np.mean(results)
        print(f"The logical error for ep = {ep} is {mean_error}")
        standard_error = np.sqrt(mean_error * (1 - mean_error) / trials)

        lower = max(0, mean_error - z * standard_error)
        upper = min(1, mean_error + z * standard_error)

        means.append(mean_error)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    return means, lower_bounds, upper_bounds


# Algorithm 2
def find_Z_code_distance(Z_logical_operators: list, periodic_graph: nx.Graph):
    distance = 100
    for logical_operator in Z_logical_operators:
        Gi = nx.Graph()
        operator_len = 100
        operator_vertices = set()
        for u in periodic_graph.nodes():
            Gi.add_node(u)
            Gi.add_node((u, 1))
        for (u, v) in periodic_graph.edges():
            if tuple(sorted((u, v))) in logical_operator:
                Gi.add_edges_from([(u, (v, 1)), ((u, 1), v)])
                operator_vertices.add(u)
                operator_vertices.add(v)
            else:
                Gi.add_edges_from([(u, v), ((u, 1), (v, 1))])
        for v in operator_vertices:
            start = v
            end = (start, 1)
            length = nx.shortest_path_length(Gi, start, end)
            if length < operator_len:
                operator_len = length

        if operator_len < distance:
            distance = operator_len
    return distance


def find_X_code_distance(X_logical_operators: list, Z_logical_operators: list):
    distance = 100
    for X_logical in X_logical_operators:
        for Z_logical in Z_logical_operators:
            if len(X_logical.intersection(Z_logical)) % 2 == 1 and len(Z_logical) < distance:
                distance = len(Z_logical)
    return distance


def draw_logical_operators(G: nx.Graph, vertices_positions: dict, logicals: list, basis: str):
    """
    Draws each logical operator on a separate figure with a distinct color and index label.

    Parameters:
        G (networkx.Graph): The input graph.
        G_pos_dict (dict): Dictionary of node positions, e.g., {node: (x, y)}.
        logicals (list): A list of sets of edges (each edge is a tuple (u, v)) representing logical operators.
    """

    cmap = colormaps['tab10']  # You can change to 'tab20' or any other

    for i, logical in enumerate(logicals):
        plt.figure(figsize=(12, 12))
        ax = plt.gca()

        # Draw the base graph in light gray
        nx.draw(
            G,
            pos=vertices_positions,
            node_color="lightgray",
            edge_color="lightgray",
            node_size=10,
            with_labels=False,
            ax=ax
        )

        color = cmap(i % 10)  # Loop through color palette
        valid_edges = []

        for u, v in logical:
            edge = (u, v) if G.has_edge(u, v) else (v, u)
            if G.has_edge(*edge):
                valid_edges.append(edge)
            else:
                print(f"[Logical {i+1}] Edge {u}-{v} is not in the graph and will be skipped.")

        # Draw the logical operator
        nx.draw_networkx_edges(
            G,
            pos=vertices_positions,
            edgelist=valid_edges,
            edge_color=[color] * len(valid_edges),
            width=6,
            ax=ax
        )
        if basis == 'X':
            # Add legend and title
            legend = [Line2D([0], [0], color=color, lw=2.5, label=f"X_Logical {i + 1}")]
            plt.title(f"X_Logical Operator {i + 1}")
        elif basis == 'Z':
            legend = [Line2D([0], [0], color=color, lw=2.5, label=f"Z_Logical {i + 1}")]
            plt.title(f"Z_Logical Operator {i + 1}")
        else:
            raise ValueError(f"Unknown basis {basis}")
        plt.legend(handles=legend, loc="upper right")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    trials = 10000
    p = 8
    q = 3
    p_B = 8
    q_B = 8

    CT_matrices_dic = {

        9: [ [  2,  3,  1,  6,  9,  8,  5,  4,  7 ],
              [  3,  1,  2,  8,  7,  4,  9,  6,  5 ],
              [  4,  6,  8,  9,  1,  7,  3,  5,  2 ],
              [  5,  9,  7,  1,  8,  2,  6,  3,  4 ],
              [  6,  8,  4,  7,  2,  5,  1,  9,  3 ],
              [  7,  5,  9,  3,  6,  1,  4,  2,  8 ],
              [  8,  4,  6,  5,  3,  9,  2,  7,  1 ],
              [  9,  7,  5,  2,  4,  3,  8,  1,  6 ] ],

        12: [[2, 10, 1, 6, 11, 9, 5, 7, 12, 4, 3, 8],
             [3, 1, 11, 10, 7, 4, 8, 12, 6, 2, 5, 9],
             [4, 6, 10, 12, 1, 8, 3, 11, 7, 9, 2, 5],
             [5, 11, 7, 1, 12, 2, 9, 6, 10, 3, 8, 4],
             [6, 9, 4, 8, 2, 7, 1, 3, 5, 12, 10, 11],
             [7, 5, 8, 3, 9, 1, 6, 4, 2, 11, 12, 10],
             [8, 7, 12, 11, 6, 3, 4, 10, 1, 5, 9, 2],
             [9, 12, 6, 7, 10, 5, 2, 1, 11, 8, 4, 3]],

        16: [[2, 10, 1, 5, 6, 7, 4, 11, 12, 3, 15, 16, 8, 9, 13, 14],
             [3, 1, 10, 7, 4, 5, 6, 13, 14, 2, 8, 9, 15, 16, 11, 12],
             [4, 5, 7, 3, 1, 2, 10, 12, 15, 6, 16, 13, 9, 11, 14, 8],
             [5, 6, 4, 1, 2, 10, 3, 16, 13, 7, 14, 8, 12, 15, 9, 11],
             [6, 7, 5, 2, 10, 3, 1, 14, 8, 4, 9, 11, 16, 13, 12, 15],
             [7, 4, 6, 10, 3, 1, 2, 9, 11, 5, 12, 15, 14, 8, 16, 13],
             [8, 11, 13, 12, 16, 14, 9, 6, 1, 15, 7, 2, 5, 3, 4, 10],
             [9, 12, 14, 15, 13, 8, 11, 1, 7, 16, 2, 4, 3, 6, 10, 5]],

        25: [[2, 10, 1, 11, 12, 13, 14, 15, 16, 17, 6, 19, 18, 5, 22, 23, 3, 4, 7, 8, 9, 25, 24, 21, 20],
             [3, 1, 17, 18, 14, 11, 19, 20, 21, 2, 4, 5, 6, 7, 8, 9, 10, 13, 12, 25, 24, 15, 16, 23, 22],
             [4, 11, 18, 8, 1, 22, 17, 23, 5, 6, 15, 2, 25, 3, 24, 12, 13, 20, 10, 16, 14, 21, 19, 7, 9],
             [5, 12, 14, 1, 9, 10, 24, 4, 25, 19, 2, 16, 17, 21, 11, 20, 7, 3, 23, 18, 22, 6, 8, 15, 13],
             [6, 13, 11, 22, 10, 20, 1, 21, 19, 18, 25, 17, 8, 2, 9, 7, 4, 15, 3, 24, 12, 16, 14, 5, 23],
             [7, 14, 19, 17, 24, 1, 16, 13, 15, 5, 3, 21, 2, 23, 18, 22, 12, 10, 9, 6, 8, 4, 25, 20, 11],
             [8, 15, 20, 23, 4, 21, 13, 19, 1, 22, 24, 11, 9, 18, 7, 2, 25, 16, 6, 12, 3, 14, 10, 17, 5],
             [9, 16, 21, 5, 25, 19, 15, 1, 13, 23, 12, 20, 7, 22, 2, 18, 24, 14, 8, 3, 6, 10, 4, 11, 17]],

        # NSG[61332]
        35: [[2, 10, 1, 11, 12, 13, 14, 15, 16, 25, 35, 20, 34, 18, 29, 33, 3, 4, 5, 6, 7, 8, 9, 32, 28, 23, 26, 17,
              24, 27, 22, 31, 30, 19, 21],
             [3, 1, 17, 18, 19, 20, 21, 22, 23, 2, 4, 5, 6, 7, 8, 9, 28, 14, 34, 12, 35, 31, 26, 29, 10, 27, 30, 25,
              15, 33, 32, 24, 16, 13, 11],
             [4, 11, 18, 24, 1, 25, 8, 26, 6, 35, 32, 2, 28, 15, 23, 13, 14, 29, 3, 10, 22, 27, 20, 16, 21, 12, 5,
              7, 9, 19, 30, 33, 34, 17, 31],
             [5, 12, 19, 1, 27, 9, 28, 7, 29, 20, 2, 26, 16, 17, 14, 24, 34, 3, 30, 23, 25, 21, 15, 4, 6, 8, 22, 13,
              18, 31, 35, 11, 32, 33, 10],
             [6, 13, 20, 25, 9, 30, 1, 4, 31, 34, 28, 16, 27, 2, 11, 22, 12, 10, 23, 33, 3, 18, 32, 21, 19, 24, 29,
              5, 35, 15, 14, 7, 8, 26, 17],
             [7, 14, 21, 8, 28, 1, 32, 33, 5, 18, 15, 17, 2, 31, 30, 12, 35, 22, 25, 3, 24, 16, 19, 26, 4, 34, 13,
              11, 27, 6, 9, 23, 20, 10, 29],
             [8, 15, 22, 26, 7, 4, 33, 34, 1, 29, 23, 14, 11, 30, 19, 2, 31, 27, 21, 18, 16, 13, 3, 12, 24, 17, 28,
              32, 5, 25, 6, 20, 10, 35, 9],
             [9, 16, 23, 6, 29, 31, 5, 1, 35, 33, 13, 24, 22, 12, 2, 21, 26, 20, 15, 32, 19, 3, 11, 25, 30, 4, 18,
              27, 10, 14, 17, 28, 7, 8, 34]]

    }

    error_probabilities = [0.002 * j for j in range(25)]
    all_params = []
    results_dict = {}
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    markers = ['o', 's', 'D', '^', 'v', '>', '<']

    for idx, N in enumerate(CT_matrices_dic.keys()):
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        # Algorithm 1 Steps 1-3
        G_vertices = generate_vertices(p, q, p_B, q_B, N)

        # Algorithm 1 Step 3 (part 2)
        G, adj_G, G_vertices_to_edges, G_edges_to_vertices, G_pos_dict = generate_hyperbolic_graph(G_vertices, draw=False)

        # Algorithm 1 Step 4
        sparse_matrix = create_adjacency_matrix(N, CT_matrices_dic[N], p_B, p, q)

        # Algorithm 1 Step 5
        periodic_G = add_periodicity_edges(G, G_vertices_to_edges, G_edges_to_vertices, sparse_matrix, G_pos_dict, draw=False)

        # Algorithm 3
        genus = surface_genus(periodic_G, p)

        print(f"The genus is {genus}")

        HCB, original_p_cycles, HCB_faces, intersection_edges, X_logicals = HCB_Algorithm(original_graph=G,
                                                                                          periodic_graph=periodic_G,
                                                                                          genus=genus, p=p)

        original_dual_G = generate_dual_graph(faces=original_p_cycles,
                                              intersection_edges=intersection_edges,
                                              p=p,
                                              vertices_to_edges=G_vertices_to_edges,
                                              G_pos_dict=G_pos_dict,
                                              draw=False)

        periodic_dual_G = generate_dual_graph(faces=HCB_faces,
                                              intersection_edges=intersection_edges,
                                              p=p,
                                              vertices_to_edges=G_vertices_to_edges,
                                              G_pos_dict=G_pos_dict,
                                              draw=False)

        dual_HCB, dual_original_faces, dual_faces, dual_intersection_edges, dual_logicals = (
            HCB_Algorithm(original_graph=original_dual_G,
                          periodic_graph=periodic_dual_G,
                          genus=genus,
                          p=q,
                          dual=True))

        Z_logicals = find_Z_logicals(X_logicals=X_logicals,
                                     potential_Z_logicals=dual_logicals,
                                     intersection_edges=intersection_edges)


        n = periodic_G.number_of_edges()
        k = 2 * (N + 1)
        encoding_rate = k / n
        d_Z = find_Z_code_distance(Z_logicals, periodic_G)
        print("d_Z is", d_Z)

        d_X = find_X_code_distance(X_logicals, Z_logicals)
        print(f"d_X is", d_X)

        all_params.append((n,k,d_Z))

        # Now returns 3 values
        means, lower_bounds, upper_bounds = error_graph(
            periodic_G, G_vertices_to_edges, error_probabilities, X_logicals, trials)

        means = np.array(means)
        results_dict[N] = means

        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        yerr = np.vstack([means - lower_bounds, upper_bounds - means])

        plt.errorbar(
            error_probabilities, means,
            yerr=yerr,
            fmt=marker, color=color,
            label=fr'$[[n={n},\ k={k},\ d_X={d_X}]$',
            capsize=5,
            linestyle='-'
        )


    plt.xlabel('Physical Error Probability')
    plt.ylabel('Logical Error Probability')
    plt.title(f'{{{p},{q}}} HQECC Error Threshold Graph', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.show()
