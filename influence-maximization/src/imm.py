"""
IMM (Influence Maximization via Martingale / Reverse Influence Sampling) Module
Implements the highly efficient RIS-based IMM algorithm for large-scale social networks.
"""

import time
import random
from typing import List, Tuple, Set, Dict
import networkx as nx

# Constants
DEFAULT_K = 10
DEFAULT_NUM_RR_SETS = 10000

def generate_rr_set(G: nx.DiGraph, node: int) -> Set[int]:
    """
    Generates a single Reverse Reachable (RR) set starting from a target node.
    Performs a reverse BFS cascade, traversing incoming edges backwards with
    their propagation probabilities.

    Args:
        G (nx.DiGraph): The social network directed graph.
        node (int): The target node to start the reverse cascade from.

    Returns:
        Set[int]: The set of nodes that can reach the target node in this sample.
    """
    # Visited nodes representing the RR set
    visited: Set[int] = {node}
    queue: List[int] = [node]

    # Reverse BFS propagation
    while queue:
        curr = queue.pop(0)
        # predecessors(curr) retrieves nodes v with directed edges (v -> curr)
        for predecessor in G.predecessors(curr):
            if predecessor not in visited:
                # Retrieve propagation probability on edge predecessor -> curr
                prob = G[predecessor][curr].get('prob', 0.0)
                
                # Flip a coin for activation
                if random.random() < prob:
                    visited.add(predecessor)
                    queue.append(predecessor)

    return visited

def imm_influence_maximization(
    G: nx.DiGraph, 
    k: int = DEFAULT_K, 
    num_rr_sets: int = DEFAULT_NUM_RR_SETS
) -> Tuple[List[int], List[float], float]:
    """
    Finds k seed nodes using the Reverse Influence Sampling (RIS) framework.

    Args:
        G (nx.DiGraph): The social network directed graph.
        k (int): Number of seeds to choose.
        num_rr_sets (int): Number of random RR sets to generate.

    Returns:
        Tuple[List[int], List[float], float]:
            - List of selected seeds (length k).
            - List of estimated influence coverage progress (length k).
            - Total runtime in seconds.
    """
    start_time = time.time()
    num_nodes = G.number_of_nodes()
    
    if num_nodes == 0 or k == 0:
        return [], [], 0.0

    print(f"\nStarting IMM (Reverse Influence Sampling) Optimization (k={k}, RR sets={num_rr_sets})")

    # Step 1: Generate random RR sets
    # We choose a random node uniformly, then construct its RR set
    all_nodes = list(G.nodes())
    rr_sets: List[Set[int]] = []
    
    # Pre-generate RR sets
    for _ in range(num_rr_sets):
        target_node = random.choice(all_nodes)
        rr_sets.append(generate_rr_set(G, target_node))

    # Step 2: Build a map from node to the indices of RR sets that contain it
    # This allows constant-time checks and rapid greedy selection
    node_to_rr: Dict[int, Set[int]] = {node: set() for node in G.nodes()}
    for rr_idx, rr in enumerate(rr_sets):
        for node in rr:
            node_to_rr[node].add(rr_idx)

    # Step 3: Greedily pick k nodes that cover the most RR sets
    seeds: List[int] = []
    covered_rr_indices: Set[int] = set()
    influence_history: List[float] = []

    # To optimize greedy lookup, maintain the current coverage lengths
    # We will dynamically update counts
    node_cover_counts = {node: len(node_to_rr[node]) for node in G.nodes()}

    for round_num in range(1, k + 1):
        best_node = -1
        best_coverage = -1

        # Locate the node covering the most uncovered RR sets
        # (This is equivalent to the maximum cover problem solver)
        for node in G.nodes():
            if node in seeds:
                continue
            
            # Since some connected RR sets may have been covered already, we could 
            # recalculate or count accurately. For maximum accuracy:
            uncovered_count = len(node_to_rr[node] - covered_rr_indices)
            if uncovered_count > best_coverage:
                best_coverage = uncovered_count
                best_node = node

        if best_node == -1:
            break

        # Record seed and cover their RR sets
        seeds.append(best_node)
        newly_covered = node_to_rr[best_node] - covered_rr_indices
        covered_rr_indices.update(newly_covered)

        # Estimate expected influence coverage
        # Formula: N * (fraction of covered RR sets)
        fraction_covered = len(covered_rr_indices) / num_rr_sets
        estimated_influence = fraction_covered * num_nodes
        influence_history.append(estimated_influence)

        round_time = time.time() - start_time
        print(f"  [IMM] Selected Seed {round_num}/{k}: Node {best_node} | "
              f"Cumulative Est Influence: {estimated_influence:.2f} | Time Elapsed: {round_time:.1f}s")

    total_runtime = time.time() - start_time
    print(f"IMM optimization completed in {total_runtime:.2f} seconds.")

    return seeds, influence_history, total_runtime
