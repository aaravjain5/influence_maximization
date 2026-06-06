"""
Independent Cascade (IC) Model Module
Simulates information diffusion process across a social network starting from a seed set.
"""

import random
from typing import Set, List
import networkx as nx
from tqdm import tqdm

# Constants
DEFAULT_NUM_SIMULATIONS = 1000

def simulate_independent_cascade(
    G: nx.DiGraph, 
    seeds: List[int], 
    num_simulations: int = DEFAULT_NUM_SIMULATIONS
) -> float:
    """
    Simulates the Independent Cascade (IC) process starting from a set of seed nodes
    to estimate the expected total number of influenced nodes.

    Args:
        G (nx.DiGraph): The social network directed graph. Edges must have 'prob' attributes.
        seeds (List[int]): The list of initial activated nodes (the seed set).
        num_simulations (int): Number of Monte Carlo simulations to run for expected value.

    Returns:
        float: The average number of active nodes at the end of the diffusion.
    """
    if not seeds:
        return 0.0

    total_activated_nodes = 0

    # Execute simulation rounds (Monte Carlo simulations)
    for _ in tqdm(range(num_simulations), desc="Simulating Independent Cascade propagation"):
        # Set of active nodes for the current simulation
        active = set(seeds)
        
        # Newly activated nodes in the previous step, which try to activate their neighbors now
        newly_active = list(seeds)

        # BFS cascade propagation
        while newly_active:
            next_active = []
            for u in newly_active:
                # Attempt to activate outgoing neighbors
                for v in G.neighbors(u):
                    if v not in active:
                        # Fetch propagation probability from edge
                        prob = G[u][v].get('prob', 0.0)
                        
                        # Coin flip simulation for activation check
                        if random.random() < prob:
                            active.add(v)
                            next_active.append(v)
            newly_active = next_active

        # Accumulate the final number of active nodes
        total_activated_nodes += len(active)

    # Return average active nodes across all simulations
    return total_activated_nodes / num_simulations
