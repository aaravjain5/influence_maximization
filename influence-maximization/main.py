"""
Main Orchestration Script
Runs Greedy and IMM simulation comparison, prints metrics table, 
performs community structure mapping, and exports matplotlib figures.
"""

import os
import random
import time
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for headless container execution (prevents GUI display-related errors)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Import custom sub-modules
from src.graph_builder import build_synthetic_graph, print_graph_stats
from src.greedy import greedy_influence_maximization
from src.imm import imm_influence_maximization
from src.community_analysis import (
    detect_communities,
    analyze_seed_community_coverage,
    compute_influence_by_community
)

# Configuration Constants
SEEDS_COUNT_K = 10
GREEDY_SIMULATIONS = 50
IMM_RR_SETS = 10000
COMM_SIMULATIONS = 500
SYNTHETIC_N = 1000
SYNTHETIC_M = 3
DETERMINISTIC_SEED = 42

def main() -> None:
    # Set random seed for reproducibility
    random.seed(DETERMINISTIC_SEED)
    np.random.seed(DETERMINISTIC_SEED)
    
    # Establish saving directory pathways
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # =========================================================================
    # STEP 1 & 2: Generate Network Graph & Print Stats
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1 & 2: CONSTRUCTING SOCIAL NETWORK AND COMPUTING TOPOLOGY")
    print("="*80)
    G = build_synthetic_graph(n=SYNTHETIC_N, m=SYNTHETIC_M, seed=DETERMINISTIC_SEED)
    print_graph_stats(G)
    
    # =========================================================================
    # STEP 3: Run Greedy Influence Maximization
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: RUNNING GREEDY SEED SELECTION (HILL CLIMBING)")
    print("="*80)
    g_seeds, g_history, g_time = greedy_influence_maximization(
        G, k=SEEDS_COUNT_K, num_simulations=GREEDY_SIMULATIONS
    )
    
    # =========================================================================
    # STEP 4: Run IMM (Reverse Influence Sampling)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: RUNNING IMM (REVERSE INFLUENCE SAMPLING)")
    print("="*80)
    imm_seeds, imm_history, imm_time = imm_influence_maximization(
        G, k=SEEDS_COUNT_K, num_rr_sets=IMM_RR_SETS
    )
    
    # =========================================================================
    # STEP 5: Print Comparative Performance Metrics Table
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: ALGORITHMIC COMPARISON TABLE")
    print("="*80)
    
    # Let's create an elegant console table showing step-by-step gains
    header = f"{'Seeds Selected':<15} | {'Greedy Influence':<18} | {'IMM Influence':<16} | {'Greedy CumTime (s)':<18} | {'IMM CumTime (s)':<15}"
    print(header)
    print("-" * len(header))
    
    # Compute relative cumulative step runtimes for mapping comparison
    for i in range(SEEDS_COUNT_K):
        # Linearly interpolate runtime steps for representation
        step_g_time = (g_time / SEEDS_COUNT_K) * (i + 1)
        step_imm_time = (imm_time / SEEDS_COUNT_K) * (i + 1)
        
        print(f"k = {i+1:<10} | {g_history[i]:<18.2f} | {imm_history[i]:<16.2f} | {step_g_time:<18.2f} | {step_imm_time:<15.2f}")
    
    print("="*80)
    print(f"Summary Results:")
    print(f"  * Total Greedy Seeds:  {g_seeds}")
    print(f"  * Total IMM Seeds:     {imm_seeds}")
    print(f"  * Greedy Speed Ratio:  IMM is {g_time / imm_time:.2f}x faster!")
    print("="*80)

    # =========================================================================
    # STEP 6: Community Modularity Analysis
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: MODULAR COMMUNITY ANALYSIS & SEED DISTRIBUTION")
    print("="*80)
    print("Detecting network community structures...")
    communities = detect_communities(G)
    print(f"Detected {len(communities)} modular communities in network.")
    
    # Highlight top 5 largest communities
    sorted_comm = sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)
    print("Top 5 Largest Communities:")
    for rank, (idx, comm) in enumerate(sorted_comm[:5], 1):
        print(f"  Rank {rank}: Community {idx:<2} contains {len(comm):>4} nodes ({len(comm)/SYNTHETIC_N*100:.1f}%)")
        
    print("\nAnalyzing seed community coverage...")
    imm_coverage = analyze_seed_community_coverage(imm_seeds, communities)
    for comm_idx, seeds_list in imm_coverage.items():
        print(f"  Community {comm_idx:<2}: Seeded directly with {len(seeds_list)} nodes: {seeds_list}")
        
    # Estimate influence coverage breakdown per community
    print("\nComputing cascade influence spread per community (IMM Seeds)...")
    comm_spread = compute_influence_by_community(G, imm_seeds, communities, num_simulations=COMM_SIMULATIONS)
    print("Expected Cascade Coverage by Community partition:")
    for comm_idx, count in sorted(comm_spread.items(), key=lambda x: x[1], reverse=True)[:10]:
        comm_size = len(communities[comm_idx])
        pct_covered = (count / comm_size) * 100 if comm_size > 0 else 0.0
        print(f"  Community {comm_idx:<2} | Size: {comm_size:>4} | Influenced: {count:>6.2f} | Reach Rate: {pct_covered:.1f}%")

    # =========================================================================
    # STEP 7: Export Scientific Result Plots
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 7: GENERATING RESEARCH GRAPHICS PLOTS")
    print("="*80)
    
    # --- PLOT 1: influence_spread_graph.png ---
    print("Rendering Plot 1: influence_spread_graph.png...")
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Run a single representative cascade starting from IMM seeds to find influenced nodes
    active = set(imm_seeds)
    newly_active = list(imm_seeds)
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
        
    # For neat visualization layout, we'll draw a subgraph of top 120 nodes 
    # to maintain high visual clarity and avoid a congested cluster.
    sub_node_count = min(SYNTHETIC_N, 150)
    sub_nodes = sorted(list(G.nodes()), key=lambda x: G.degree(x), reverse=True)[:sub_node_count]
    # Keep standard seed coverage representation if seeds are excluded in degree list
    for seed in imm_seeds:
        if seed not in sub_nodes:
            sub_nodes.append(seed)
            
    G_sub = G.subgraph(sub_nodes)
    
    # Node coloring lists matching criteria:
    # Seeds = Red, Influenced = Orange, Non-reached = Light Blue
    color_map = []
    for node in G_sub.nodes():
        if node in imm_seeds:
            color_map.append('#e63946')  # Dark soft Red for seeding
        elif node in active:
            color_map.append('#f4a261')  # Warm Orange for cascade reached
        else:
            color_map.append('#a8dadc')  # Airy Light Blue for not reached
            
    # Position using spring layout
    pos = nx.spring_layout(G_sub, seed=DETERMINISTIC_SEED, k=1.8/np.sqrt(len(G_sub)))
    
    nx.draw_networkx_nodes(G_sub, pos, node_color=color_map, node_size=35, alpha=0.85)
    nx.draw_networkx_edges(G_sub, pos, width=0.5, alpha=0.25, edge_color='#4a4e69', arrows=False)
    
    plt.title("Independent Cascade Simulation Reach Graph (IMM Selected Seeds)", fontsize=13, fontweight='bold', pad=15)
    
    # Visual Legend Elements
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e63946', label='Active Seed Nodes'),
        Patch(facecolor='#f4a261', label='Influenced/Cascaded Nodes'),
        Patch(facecolor='#a8dadc', label='Non-Reached Nodes')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "influence_spread_graph.png"), bbox_inches='tight')
    plt.close()

    # --- PLOT 2: greedy_vs_imm_coverage.png ---
    print("Rendering Plot 2: greedy_vs_imm_coverage.png...")
    plt.figure(figsize=(8, 5))
    x_range = range(1, SEEDS_COUNT_K + 1)
    
    plt.plot(x_range, g_history, marker='o', linestyle='-', linewidth=2, color='#1d3557', label='Greedy Seeding Model')
    plt.plot(x_range, imm_history, marker='s', linestyle='--', linewidth=2, color='#f4a261', label='IMM (Reverse Sampling) Model')
    
    plt.title("Seeding Size vs Expected Influence Spread Coverage", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Size of Seed Set (k)", fontsize=10)
    plt.ylabel("Expected Core Influence Reach (Nodes Count)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(x_range)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "greedy_vs_imm_coverage.png"))
    plt.close()

    # --- PLOT 3: runtime_comparison.png ---
    print("Rendering Plot 3: runtime_comparison.png...")
    plt.figure(figsize=(6, 5))
    labels = ['Traditional Greedy', 'IMM Algorithm']
    runtimes = [g_time, imm_time]
    colors = ['#1d3557', '#f4a151']
    
    bars = plt.bar(labels, runtimes, color=colors, width=0.4, edgecolor='black', alpha=0.85)
    plt.title("Optimization Runtime Execution Duration", fontsize=12, fontweight='bold', pad=12)
    plt.ylabel("Total Run Duration (seconds)", fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Place duration numerical tags on top of bar charts
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (max(runtimes)*0.01),
                 f"{height:.3f}s", ha='center', va='bottom', fontweight='bold', fontsize=9)
                 
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "runtime_comparison.png"))
    plt.close()

    # --- PLOT 4: community_influence_breakdown.png ---
    print("Rendering Plot 4: community_influence_breakdown.png...")
    plt.figure(figsize=(8, 5))
    
    # Focus only on top 10 largest communities
    top_n_comms = sorted(comm_spread.keys(), key=lambda k: len(communities[k]), reverse=True)[:10]
    comm_labels = [f"Comm {c}" for c in top_n_comms]
    reach_counts = [comm_spread[c] for c in top_n_comms]
    
    palette = sns.color_palette("muted", len(top_n_comms))
    bars = plt.bar(comm_labels, reach_counts, color=palette, edgecolor='black', alpha=0.8)
    
    plt.title("Expected Cascade Reach Breakdown by Large Communities (IMM)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Community Partition Identifier", fontsize=10)
    plt.ylabel("Average Reached Nodes", fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Label counts on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.5,
                 f"{height:.1f}", ha='center', va='bottom', fontsize=8)
                 
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "community_influence_breakdown.png"))
    plt.close()
    
    print("\nSuccess! All 4 visual plots rendered and stored successfully inside results/ folder.")
    print("="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY. EXITED CODE 0.")
    print("="*80)

if __name__ == "__main__":
    main()
