"""
Graph Builder Module
Generates synthetic social network structures or loads real edge list networks,
enriching them with random propagation probabilities.
"""

import os
import random
from typing import Set, Dict, Any
import networkx as nx

# Define Global Constants
DEFAULT_SECTOR_SEED = 42
MIN_PROP_PROB = 0.01
MAX_PROP_PROB = 0.10

def build_synthetic_graph(n: int = 1000, m: int = 3, seed: int = DEFAULT_SECTOR_SEED) -> nx.DiGraph:
    """
    Generates a Barabasi-Albert (scale-free) graph and converts it to a directed
    graph representing a social network with random edge propagation probabilities.

    Args:
        n (int): Number of nodes.
        m (int): Number of edges to attach from a new node to existing nodes.
        seed (int): Random seed for reproducibility.

    Returns:
        nx.DiGraph: A directed network with edge propagation probabilities under the 'prob' key.
    """
    # Fix seed for reproducibility
    random.seed(seed)
    
    # Scale-free networks are representative of social networks
    G_undirected = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    
    # Social network influences are typically directional, convert undirected to directed
    G = G_undirected.to_directed()
    
    # Assign edge influence probability probabilities randomly
    for u, v in G.edges():
        G[u][v]['prob'] = random.uniform(MIN_PROP_PROB, MAX_PROP_PROB)
        
    return G

def load_real_graph(filepath: str) -> nx.DiGraph:
    """
    Loads a network from an edgelist file (tab or space-separated) and assigns 
    random edge influence probabilities.

    Args:
        filepath (str): Path to the edgelist file.

    Returns:
        nx.DiGraph: A directed network with edge propagation probabilities.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Selected graph file not found at: {filepath}")
        
    # Read directed graph from file
    G = nx.read_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int)
    
    # Assign edge influence probability probabilities randomly
    for u, v in G.edges():
        G[u][v]['prob'] = random.uniform(MIN_PROP_PROB, MAX_PROP_PROB)
        
    return G

def print_graph_stats(G: nx.DiGraph) -> None:
    """
    Prints descriptive statistics of the social network.

    Args:
        G (nx.DiGraph): The social network graph to describe.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Calculate average degree
    degrees = [G.degree(node) for node in G.nodes()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0.0
    density = nx.density(G)
    
    # Connected components require undirected representation
    G_undir = G.to_undirected()
    num_cc = nx.number_connected_components(G_undir)
    
    # Find influence leaders by getting top 5 degree centrality nodes
    deg_centrality = nx.degree_centrality(G)
    top_centrality = sorted(deg_centrality.items(), key=lambda item: item[1], reverse=True)[:5]
    
    print("=" * 60)
    print("SOCIAL NETWORK GRAPH ANALYSIS & STATS")
    print("=" * 60)
    print(f"Number of Nodes:                {num_nodes:,}")
    print(f"Number of Directed Edges:       {num_edges:,}")
    print(f"Average Degree (In + Out):      {avg_degree:.4f}")
    print(f"Graph Density:                  {density:.6f}")
    print(f"Connected Components (Undir):   {num_cc}")
    print("-" * 60)
    print("Top 5 Nodes by Degree Centrality:")
    for rank, (node, score) in enumerate(top_centrality, 1):
        degree = G.degree(node)
        print(f"  {rank}. Node {node:<5} | Centrality Score: {score:.4f} | Absolute Degree: {degree}")
    print("=" * 60)
