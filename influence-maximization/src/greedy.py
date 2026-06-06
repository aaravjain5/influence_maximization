"""
Greedy Influence Maximization Module
Implements the baseline hill-climbing Greedy algorithm for selecting a set of k seeds
maximizing expected influence cascade.
"""

import time
from typing import List, Tuple, Set
import networkx as nx
from src.ic_model import simulate_independent_cascade

# Constants
DEFAULT_K = 10
DEFAULT_SIMULATIONS = 300

def greedy_influence_maximization(
    G: nx.DiGraph, 
    k: int = DEFAULT_K, 
    num_simulations: int = DEFAULT_SIMULATIONS
) -> Tuple[List[int], List[float], float]:
    """
    Finds k seed nodes using a greedy strategy of maximizing marginal influence gain.

    Args:
        G (nx.DiGraph): The social network directed graph.
        k (int): Number of seed nodes to select.
        num_simulations (int): Number of Monte Carlo simulations to evaluate influence at each step.

    Returns:
        Tuple[List[int], List[float], float]: 
            - List of selected seed nodes (length k).
            - List of expected influence coverage after picking each seed (length k).
            - Total runtime in seconds.
    """
    start_time = time.time()
    seeds: List[int] = []
    influence_history: List[float] = []

    print(f"\nStarting Greedy Influence Maximization (k={k}, simulations={num_simulations})")
    
    for round_num in range(1, k + 1):
        best_candidate = -1
        best_avg_influence = -1.0
        
        # Test every non-seed node for marginal gain
        for node in G.nodes():
            if node in seeds:
                continue
                
            # Form candidate seed list
            candidate_seeds = seeds + [node]
            
            # Estimate candidate's influence spread
            # Note: num_simulations is kept moderate for greedy due to O(k * N * R) time complexity
            avg_influence = simulate_independent_cascade(G, candidate_seeds, num_simulations=num_simulations)
            
            if avg_influence > best_avg_influence:
                best_avg_influence = avg_influence
                best_candidate = node
                
        # Commit the best candidate
        seeds.append(best_candidate)
        influence_history.append(best_avg_influence)
        
        round_time = time.time() - start_time
        print(f"  [Greedy] Selected Seed {round_num}/{k}: Node {best_candidate} | "
              f"Cumulative Influence: {best_avg_influence:.2f} | Time Elapsed: {round_time:.1f}s")

    total_runtime = time.time() - start_time
    print(f"Greedy optimization completed in {total_runtime:.2f} seconds.")
    
    return seeds, influence_history, total_runtime
