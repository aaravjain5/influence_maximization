"""
Community Analysis Module
Detects network community structures using modularity maximization,
and analyzes seed spread and cascade coverage across distinct communities.
"""

import random
from typing import List, Dict, Set, Any
import networkx as nx
from src.ic_model import simulate_independent_cascade

# Constants
DEFAULT_COMM_SIMULATIONS = 500

def detect_communities(G: nx.DiGraph) -> List[Set[int]]:
    """
    Detects communities in the social network using greedy modularity optimization.
    Converts to an undirected graph to ensure standard modularity calculations.

    Args:
        G (nx.DiGraph): The social network directed graph.

    Returns:
        List[Set[int]]: A list of community node sets.
    """
    G_undir = G.to_undirected()
    communities_generator = nx.community.greedy_modularity_communities(G_undir)
    
    # Convert generator/frozensets list to standard mutable sets lists
    communities = [set(comm) for comm in communities_generator]
    return communities

def analyze_seed_community_coverage(seeds: List[int], communities: List[Set[int]]) -> Dict[int, List[int]]:
    """
    Analyzes which communities are directly seeded by the chosen seed set.

    Args:
        seeds (List[int]): The chosen seed set nodes.
        communities (List[Set[int]]): List of community node sets.

    Returns:
        Dict[int, List[int]]: Map from community index to list of seed nodes inside it.
    """
    coverage: Dict[int, List[int]] = {}
    
    # Check each seed's membership across communities
    for seed in seeds:
        for comm_idx, community in enumerate(communities):
            if seed in community:
                if comm_idx not in coverage:
                    coverage[comm_idx] = []
                coverage[comm_idx].append(seed)
                break  # Each node belongs to exactly one partition community
                
    return coverage

def compute_influence_by_community(
    G: nx.DiGraph, 
    seeds: List[int], 
    communities: List[Set[int]], 
    num_simulations: int = DEFAULT_COMM_SIMULATIONS
) -> Dict[int, float]:
    """
    Runs Independent Cascade simulations and calculates the average number
    of influenced nodes inside each community.

    Args:
        G (nx.DiGraph): The directed social network graph.
        seeds (List[int]): The seed nodes.
        communities (List[Set[int]]): List of community node sets.
        num_simulations (int): Number of Monte Carlo trials to average coverage.

    Returns:
        Dict[int, float]: Map from community index to expected number of activated nodes.
    """
    if not seeds or not communities:
        return {}

    # Initialize average influence counts per community
    community_influence: Dict[int, float] = {i: 0.0 for i in range(len(communities))}
    
    # Create a quick-lookup map from node ID to community index for speedy simulation matches
    node_to_comm: Dict[int, int] = {}
    for comm_idx, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = comm_idx

    # Run simulations
    for _ in range(num_simulations):
        # Local cascade state
        active: Set[int] = set(seeds)
        newly_active: List[int] = list(seeds)

        while newly_active:
            next_active = []
            for u in newly_active:
                for v in G.neighbors(u):
                    if v not in active:
                        prob = G[u][v].get('prob', 0.0)
                        if random.random() < prob:
                            active.add(v)
                            next_active.append(v)
            newly_active = next_active

        # Attribute activated nodes to communities
        for activated_node in active:
            if activated_node in node_to_comm:
                comm_idx = node_to_comm[activated_node]
                community_influence[comm_idx] += 1.0

    # Average the accumulated counts across all Monte Carlo simulations
    for comm_idx in community_influence:
        community_influence[comm_idx] /= num_simulations

    return community_influence
