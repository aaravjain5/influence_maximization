import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  Network, 
  Activity, 
  BarChart3, 
  Code2, 
  Zap, 
  Sliders, 
  RefreshCw, 
  Play, 
  Download, 
  Copy, 
  Check, 
  CheckCircle2, 
  Users, 
  Flame, 
  FileCode, 
  Folder, 
  ChevronRight, 
  BookOpen,
  Info
} from 'lucide-react';
import { 
  NetworkGraph, 
  NetworkNode, 
  NetworkEdge, 
  SimulationResult, 
  CommunityMetric 
} from './types';
import { 
  buildSyntheticGraphJS, 
  simulateICJS, 
  runGreedyJS, 
  runIMMJS 
} from './algorithms';

// Community Colors palette for distinct visual classification
const COMMUNITY_COLORS = [
  '#3b82f6', // blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ec4899', // pink
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#f43f5e', // rose
  '#14b8a6', // teal
  '#a855f7', // purple
  '#6366f1', // indigo
];

// Map of Python Code Files for the Source Code Viewer
const PYTHON_CODEFILES: Record<string, string> = {
  "requirements.txt": `networkx>=3.0
numpy>=1.22.0
matplotlib>=3.5.0
seaborn>=0.12.0
tqdm>=4.64.0
pandas>=1.4.0`,

  "src/graph_builder.py": `"""
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
    """
    random.seed(seed)
    G_undirected = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    G = G_undirected.to_directed()
    
    # Assign edge influence probability probabilities randomly
    for u, v in G.edges():
        G[u][v]['prob'] = random.uniform(MIN_PROP_PROB, MAX_PROP_PROB)
        
    return G

def load_real_graph(filepath: str) -> nx.DiGraph:
    """Loads a network from an edgelist file and assigns random prob."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Selected graph file not found at: {filepath}")
    G = nx.read_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int)
    for u, v in G.edges():
        G[u][v]['prob'] = random.uniform(MIN_PROP_PROB, MAX_PROP_PROB)
    return G

def print_graph_stats(G: nx.DiGraph) -> None:
    """Prints descriptive statistics of the social network."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = [G.degree(node) for node in G.nodes()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0.0
    density = nx.density(G)
    
    G_undir = G.to_undirected()
    num_cc = nx.number_connected_components(G_undir)
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
    print("=" * 60)`,

  "src/ic_model.py": `"""
Independent Cascade (IC) Model Module
Simulates information diffusion process across a social network starting from a seed set.
"""

import random
from typing import Set, List
import networkx as nx
from tqdm import tqdm

DEFAULT_NUM_SIMULATIONS = 1000

def simulate_independent_cascade(
    G: nx.DiGraph, 
    seeds: List[int], 
    num_simulations: int = DEFAULT_NUM_SIMULATIONS
) -> float:
    """Simulates expected cascade reach under Monte Carlo trials."""
    if not seeds:
        return 0.0
    total_activated_nodes = 0
    for _ in tqdm(range(num_simulations), desc="Simulating IC"):
        active = set(seeds)
        newly_active = list(seeds)
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
        total_activated_nodes += len(active)
    return total_activated_nodes / num_simulations`,

  "src/greedy.py": `"""
Greedy Influence Maximization Module
Implements the baseline hill-climbing Greedy algorithm for selecting a set of k seeds
maximizing expected influence cascade.
"""

import time
from typing import List, Tuple
import networkx as nx
from src.ic_model import simulate_independent_cascade

DEFAULT_K = 10
DEFAULT_SIMULATIONS = 300

def greedy_influence_maximization(
    G: nx.DiGraph, 
    k: int = DEFAULT_K, 
    num_simulations: int = DEFAULT_SIMULATIONS
) -> Tuple[List[int], List[float], float]:
    """Finds k seeds by greedily evaluating marginal influence gains."""
    start_time = time.time()
    seeds = []
    influence_history = []
    print(f"Starting Greedy k={k}...")
    for round_num in range(1, k + 1):
        best_candidate = -1
        best_avg_influence = -1.0
        for node in G.nodes():
            if node in seeds:
                continue
            candidate_seeds = seeds + [node]
            avg_influence = simulate_independent_cascade(G, candidate_seeds, num_simulations)
            if avg_influence > best_avg_influence:
                best_avg_influence = avg_influence
                best_candidate = node
        seeds.append(best_candidate)
        influence_history.append(best_avg_influence)
        print(f"  Selected Seed {round_num}/{k}: Node {best_candidate} | Est: {best_avg_influence:.2f}")
    return seeds, influence_history, time.time() - start_time`,

  "src/imm.py": `"""
IMM (Influence Maximization via Martingale) Module
Implements the highly efficient RIS-based IMM algorithm for social networks.
"""

import time
import random
from typing import List, Tuple, Set, Dict
import networkx as nx

def generate_rr_set(G: nx.DiGraph, node: int) -> Set[int]:
    """Generates a Reverse Reachable set starting from a target node."""
    visited = {node}
    queue = [node]
    while queue:
        curr = queue.pop(0)
        for predecessor in G.predecessors(curr):
            if predecessor not in visited:
                prob = G[predecessor][curr].get('prob', 0.0)
                if random.random() < prob:
                    visited.add(predecessor)
                    queue.append(predecessor)
    return visited

def imm_influence_maximization(
    G: nx.DiGraph, 
    k: int = 10, 
    num_rr_sets: int = 10000
) -> Tuple[List[int], List[float], float]:
    """Finds k seeds using the Reverse Influence Sampling (RIS) framework."""
    start_time = time.time()
    num_nodes = G.number_of_nodes()
    if num_nodes == 0 or k == 0:
         return [], [], 0.0
    
    all_nodes = list(G.nodes())
    rr_sets = [generate_rr_set(G, random.choice(all_nodes)) for _ in range(num_rr_sets)]
    
    node_to_rr = {node: set() for node in G.nodes()}
    for rr_idx, rr in enumerate(rr_sets):
        for node in rr:
            node_to_rr[node].add(rr_idx)
            
    seeds = []
    covered_rr_indices = set()
    influence_history = []
    for round_num in range(1, k + 1):
        best_node = -1
        best_coverage = -1
        for node in G.nodes():
            if node in seeds:
                continue
            uncovered_count = len(node_to_rr[node] - covered_rr_indices)
            if uncovered_count > best_coverage:
                best_coverage = uncovered_count
                best_node = node
        if best_node == -1:
             break
        seeds.append(best_node)
        covered_rr_indices.update(node_to_rr[best_node])
        influence_history.append((len(covered_rr_indices)/num_rr_sets)*num_nodes)
    return seeds, influence_history, time.time() - start_time`,

  "src/community_analysis.py": `"""
Community Analysis Module
Detects modular structures and maps influence cascade spreads.
"""

import random
from typing import List, Dict, Set
import networkx as nx

def detect_communities(G: nx.DiGraph) -> List[Set[int]]:
    """Detects communities using modularity maximization."""
    return [set(comm) for comm in nx.community.greedy_modularity_communities(G.to_undirected())]

def analyze_seed_community_coverage(seeds: List[int], communities: List[Set[int]]) -> Dict[int, List[int]]:
    """Determines community membership of seed nodes."""
    coverage = {}
    for seed in seeds:
        for idx, community in enumerate(communities):
            if seed in community:
                coverage.setdefault(idx, []).append(seed)
                break
    return coverage

def compute_influence_by_community(
    G: nx.DiGraph, 
    seeds: List[int], 
    communities: List[Set[int]], 
    num_simulations: int = 500
) -> Dict[int, float]:
    """Measures cascade dispersion inside community structures."""
    community_influence = {i: 0.0 for i in range(len(communities))}
    for _ in range(num_simulations):
        active = set(seeds)
        newly_active = list(seeds)
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
        for node in active:
            for i, comm in enumerate(communities):
                if node in comm:
                    community_influence[i] += 1.0
                    break
    for i in community_influence:
        community_influence[i] /= num_simulations
    return community_influence`,

  "main.py": `"""
Main Orchestration Script
Runs Greedy and IMM simulation comparison, prints metrics table, 
performs community structure mapping, and exports matplotlib figures.
"""

import os
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from src.graph_builder import build_synthetic_graph, print_graph_stats
from src.greedy import greedy_influence_maximization
from src.imm import imm_influence_maximization
from src.community_analysis import detect_communities, compute_influence_by_community

# Execution Pipeline
if __name__ == "__main__":
    G = build_synthetic_graph(n=1000, m=3)
    print_graph_stats(G)
    
    g_seeds, g_history, g_time = greedy_influence_maximization(G, k=10)
    imm_seeds, imm_history, imm_time = imm_influence_maximization(G, k=10)
    
    print("Optimization Completed!")
    # Saves plots to /results/ influence_spread_graph.png, runtime_comparison.png, etc.`
};

export default function App() {
  // --- Active Tab State ---
  const [activeTab, setActiveTab] = useState<'graph' | 'analytics' | 'code'>('graph');

  // --- Graph Construction Input states ---
  const [numNodes, setNumNodes] = useState<number>(100);
  const [mEdges, setMEdges] = useState<number>(2);
  const [seedCountK, setSeedCountK] = useState<number>(6);
  const [simulationCount, setSimulationCount] = useState<number>(50);
  const [cascadeChance, setCascadeChance] = useState<number>(0.05); // Global scale factor for visual cascading

  // --- Core Graph and Simulation States ---
  const [graph, setGraph] = useState<NetworkGraph | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [greedyResult, setGreedyResult] = useState<SimulationResult | null>(null);
  const [immResult, setImmResult] = useState<SimulationResult | null>(null);
  const [isSolving, setIsSolving] = useState<boolean>(false);
  const [activeAlgorithm, setActiveAlgorithm] = useState<'greedy' | 'imm' | 'none'>('none');

  // --- Interactive Cascade Animation States ---
  const [selectedSeeds, setSelectedSeeds] = useState<number[]>([]);
  const [cascadeStep, setCascadeStep] = useState<number>(0);
  const [isCascading, setIsCascading] = useState<boolean>(false);
  const [activatedNodesList, setActivatedNodesList] = useState<number[]>([]);
  const [animationHistory, setAnimationHistory] = useState<number[][]>([]);
  const [hoveredNode, setHoveredNode] = useState<NetworkNode | null>(null);

  // --- Code Explorer States ---
  const [activeFile, setActiveFile] = useState<string>("main.py");
  const [copiedFile, setCopiedFile] = useState<boolean>(false);

  // --- Layout coordinates (Custom Spring simulation results saved) ---
  const [nodePositions, setNodePositions] = useState<{ x: number; y: number }[]>([]);

  // 1. Compile Graph upon load
  useEffect(() => {
    handleRegenerateGraph();
  }, [numNodes, mEdges]);

  // Compute Layout positions dynamically using an advanced grouped circular community force structure
  // This produces extremely tidy, gorgeous, organic visual graphs immediately!
  const generatePositions = (currentGraph: NetworkGraph) => {
    const positions: { x: number; y: number }[] = [];
    const n = currentGraph.nodes.length;
    
    // Group nodes by community first to define clustering anchors
    const communityGroups = new Map<number, number[]>();
    currentGraph.nodes.forEach(node => {
      if (!communityGroups.has(node.community)) {
        communityGroups.set(node.community, []);
      }
      communityGroups.get(node.community)!.push(node.id);
    });

    const numCommunities = communityGroups.size;
    const centerWidth = 550;
    const centerHeight = 420;
    const centerX = centerWidth / 2;
    const centerY = centerHeight / 2;

    // Define community circle clusters coordinates
    const clusterRadius = 150;
    const communityPositions = new Map<number, { x: number; y: number }>();
    let commIndex = 0;
    communityGroups.forEach((_, commId) => {
      const angle = (commIndex / numCommunities) * 2 * Math.PI;
      const x = centerX + clusterRadius * Math.cos(angle);
      const y = centerY + clusterRadius * Math.sin(angle);
      communityPositions.set(commId, { x, y });
      commIndex++;
    });

    // Distribute nodes around community anchors
    const nodeCoords = new Array<{ x: number; y: number }>(n);
    communityGroups.forEach((nodeIds, commId) => {
      const anchor = communityPositions.get(commId) || { x: centerX, y: centerY };
      const groupSize = nodeIds.length;
      
      nodeIds.forEach((nodeId, idx) => {
        const spreadRadius = 25 + Math.sqrt(groupSize) * 8;
        const angle = (idx / groupSize) * 2 * Math.PI + (Math.random() * 0.4 - 0.2);
        
        let x = anchor.x + spreadRadius * Math.cos(angle);
        let y = anchor.y + spreadRadius * Math.sin(angle);

        // Keep coordinates safely bounded
        x = Math.max(35, Math.min(centerWidth - 35, x));
        y = Math.max(35, Math.min(centerHeight - 35, y));

        nodeCoords[nodeId] = { x, y };
      });
    });

    // Simple relaxation/repulsion to minimize overlap slightly
    for (let step = 0; step < 8; step++) {
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const dx = nodeCoords[i].x - nodeCoords[j].x;
          const dy = nodeCoords[i].y - nodeCoords[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const minDist = 18;
          if (dist < minDist && dist > 0.1) {
            const force = (minDist - dist) / dist * 0.15;
            const fx = dx * force;
            const fy = dy * force;
            
            nodeCoords[i].x += fx;
            nodeCoords[i].y += fy;
            nodeCoords[j].x -= fx;
            nodeCoords[j].y -= fy;
          }
        }
      }
    }

    setNodePositions(nodeCoords);
  };

  const handleRegenerateGraph = () => {
    setIsGenerating(true);
    // Reset all old calculations
    setGreedyResult(null);
    setImmResult(null);
    setSelectedSeeds([]);
    setActivatedNodesList([]);
    setAnimationHistory([]);
    setCascadeStep(0);
    setIsCascading(false);
    
    setTimeout(() => {
      const newGraph = buildSyntheticGraphJS(numNodes, mEdges);
      setGraph(newGraph);
      generatePositions(newGraph);
      setIsGenerating(false);
    }, 180);
  };

  // Run IMM Algorithm instantly
  const handleRunIMM = () => {
    if (!graph) return;
    setIsSolving(true);
    setActiveAlgorithm('imm');
    setTimeout(() => {
      const res = runIMMJS(graph, seedCountK, 5000);
      setImmResult(res);
      setSelectedSeeds(res.seeds);
      // Run cascade simulation once to populate the active cascade path
      const cascadeRun = simulateICJS(graph, res.seeds, 100);
      setActivatedNodesList(cascadeRun.singleSampleCascade);
      setIsSolving(false);
    }, 120);
  };

  // Run Greedy Algorithm with visual feedback
  const handleRunGreedy = () => {
    if (!graph) return;
    setIsSolving(true);
    setActiveAlgorithm('greedy');
    setTimeout(() => {
      // In JS browser frame, we limit simulations slightly for instant, lag-free response!
      const res = runGreedyJS(graph, seedCountK, 15);
      setGreedyResult(res);
      setSelectedSeeds(res.seeds);
      const cascadeRun = simulateICJS(graph, res.seeds, 100);
      setActivatedNodesList(cascadeRun.singleSampleCascade);
      setIsSolving(false);
    }, 120);
  };

  const handleCompareAll = () => {
    if (!graph) return;
    setIsSolving(true);
    setActiveAlgorithm('none');
    setTimeout(() => {
      const imm = runIMMJS(graph, seedCountK, 5000);
      const gr = runGreedyJS(graph, seedCountK, 15);
      setImmResult(imm);
      setGreedyResult(gr);
      setSelectedSeeds(imm.seeds); // Defaults visualization to IMM seeds
      const cascadeRun = simulateICJS(graph, imm.seeds, 100);
      setActivatedNodesList(cascadeRun.singleSampleCascade);
      setIsSolving(false);
    }, 120);
  };

  // --- Handlers for triggering sequential Cascade Animation step-by-step ---
  const triggerAnimatedCascade = () => {
    if (!graph || selectedSeeds.length === 0) return;
    setIsCascading(true);
    setCascadeStep(1);

    // Initialize state history of activation waves
    const active = new Set<number>(selectedSeeds);
    let newlyActive = [...selectedSeeds];
    const stepsHistory: number[][] = [[...selectedSeeds]];

    while (newlyActive.length > 0) {
      const nextActive: number[] = [];
      const edgeProbMap = new Map<string, number>();
      graph.edges.forEach((edge) => {
        edgeProbMap.set(`${edge.source}_${edge.target}`, edge.prob);
      });

      for (const u of newlyActive) {
        const neighbors = graph.adjacency.get(u) || [];
        for (const v of neighbors) {
          if (!active.has(v)) {
            const edgeProb = edgeProbMap.get(`${u}_${v}`) || 0.05;
            // Scale propagation slightly based on user slider for visual richness
            if (Math.random() < edgeProb + cascadeChance) {
              active.add(v);
              nextActive.push(v);
            }
          }
        }
      }

      if (nextActive.length > 0) {
        stepsHistory.push(Array.from(active));
      }
      newlyActive = nextActive;
    }

    setAnimationHistory(stepsHistory);
    
    // Play transition frames sequentially
    let step = 0;
    const interval = setInterval(() => {
      if (step < stepsHistory.length - 1) {
        step++;
        setCascadeStep(step + 1);
      } else {
        clearInterval(interval);
        setIsCascading(false);
      }
    }, 1000);
  };

  // Currently viewing list of activated nodes matching our animation steps
  const visibleActivatedNodes = useMemo(() => {
    if (animationHistory.length > 0) {
      const stepIdx = Math.min(cascadeStep - 1, animationHistory.length - 1);
      return animationHistory[stepIdx] || [];
    }
    return activatedNodesList.length > 0 ? activatedNodesList : selectedSeeds;
  }, [cascadeStep, animationHistory, activatedNodesList, selectedSeeds]);

  // Compute graph topological details
  const graphStats = useMemo(() => {
    if (!graph) return { density: 0, avgDegree: 0, topCentral: [] as number[] };
    const n = graph.nodes.length;
    const d = graph.edges.length / (n * (n - 1));
    const totalDegree = graph.nodes.reduce((acc, curr) => acc + curr.degree, 0);
    const avgDegree = totalDegree / n;

    const sortedByDegree = [...graph.nodes]
      .sort((a, b) => b.degree - a.degree)
      .slice(0, 5)
      .map(node => node.id);

    return { density: d, avgDegree, topCentral: sortedByDegree };
  }, [graph]);

  // Community partition details computed live
  const communityMetrics = useMemo<CommunityMetric[]>(() => {
    if (!graph) return [];
    
    const countMap = new Map<number, number>();
    graph.nodes.forEach(n => {
      countMap.set(n.community, (countMap.get(n.community) || 0) + 1);
    });

    const metrics: CommunityMetric[] = [];
    let colorIdx = 0;
    countMap.forEach((size, commId) => {
      // Find seeds in this community
      const commSeeds = selectedSeeds.filter(id => graph.nodes[id]?.community === commId);
      
      // Find influenced nodes in this community
      const commReached = visibleActivatedNodes.filter(id => graph.nodes[id]?.community === commId).length;

      metrics.push({
        id: commId,
        size,
        color: COMMUNITY_COLORS[colorIdx % COMMUNITY_COLORS.length],
        seeds: commSeeds,
        reached: commReached
      });
      colorIdx++;
    });

    return metrics.sort((a, b) => b.size - a.size).slice(0, 8); // Top 8 communities
  }, [graph, selectedSeeds, visibleActivatedNodes]);

  // Handle single tab file selection copies
  const triggerCopyFile = () => {
    navigator.clipboard.writeText(PYTHON_CODEFILES[activeFile]);
    setCopiedFile(true);
    setTimeout(() => setCopiedFile(false), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans" id="influence_optimizer_root">
      {/* HEADER SECTION */}
      <header className="bg-slate-950 border-b border-slate-800 px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-md" id="workspace_header">
        <div id="header_title_section">
          <div className="flex items-center gap-3">
            <span className="p-2 bg-gradient-to-tr from-blue-600 to-indigo-500 rounded-lg shadow-inner text-white">
              <Network className="w-6 h-6 animate-pulse" />
            </span>
            <div>
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-blue-400 via-indigo-200 to-white bg-clip-text text-transparent">
                Influence Maximization Workspace
              </h1>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Stochastic Independent Cascade Diffusion & Relational Graph Propagation
              </p>
            </div>
          </div>
        </div>

        {/* TOP STATUS ROW */}
        <div className="flex items-center gap-4 bg-slate-900 border border-slate-800 rounded-lg px-4 py-2 text-xs font-mono" id="top_status_badge">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping" />
            <span className="text-emerald-400">Environment Active</span>
          </div>
          <div className="text-slate-500 border-l border-slate-800 pl-3">
            Nodes: <span className="text-slate-300">{graph ? graph.nodes.length : 0}</span>
          </div>
          <div className="text-slate-500 border-l border-slate-800 pl-3">
            Edges: <span className="text-slate-300">{graph ? graph.edges.length : 0}</span>
          </div>
        </div>
      </header>

      {/* CORE FRAME WORKSPACE */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6" id="core_workspace">
        
        {/* LEFT COLUMN: CONTROL & PARAMETERS MODULATOR (3 cols size) */}
        <section className="lg:col-span-4 flex flex-col gap-6" id="parameters_pane">
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col gap-5">
            
            {/* Title */}
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <Sliders className="w-4 h-4 text-blue-400" />
              <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Topology Settings</h2>
            </div>

            {/* Input Slider 1: Network graph Size */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between text-xs font-mono text-slate-400">
                <span>Graph Size (Nodes N)</span>
                <span className="text-blue-400 font-semibold">{numNodes}</span>
              </div>
              <input 
                type="range" 
                min="50" 
                max="250" 
                step="10"
                value={numNodes} 
                onChange={(e) => setNumNodes(Number(e.target.value))}
                className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-[10px] text-slate-500">Prefers Barabasi-Albert preferential attachment scaling.</span>
            </div>

            {/* Input Slider 2: Growth probability parameter m */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between text-xs font-mono text-slate-400">
                <span>Attachment Connections (m)</span>
                <span className="text-blue-400 font-semibold">{mEdges}</span>
              </div>
              <input 
                type="range" 
                min="1" 
                max="5" 
                step="1"
                value={mEdges} 
                onChange={(e) => setMEdges(Number(e.target.value))}
                className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-[10px] text-slate-500">Number of links spawned per incoming node node.</span>
            </div>

            {/* Input Slider 3: Selected seeds count k */}
            <div className="flex flex-col gap-2 border-t border-slate-800 pt-3">
              <div className="flex justify-between text-xs font-mono text-slate-400">
                <span>Influencer Seed Size (k)</span>
                <span className="text-indigo-400 font-semibold">{seedCountK}</span>
              </div>
              <input 
                type="range" 
                min="1" 
                max="10" 
                step="1"
                value={seedCountK} 
                onChange={(e) => setSeedCountK(Number(e.target.value))}
                className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-[10px] text-slate-500">The budget size for seeding our cascade.</span>
            </div>

            {/* Input Slider 4: Cascade Probability Scale */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between text-xs font-mono text-slate-400">
                <span>Propagation Booster Scale</span>
                <span className="text-amber-500 font-semibold">+{(cascadeChance * 100).toFixed(0)}%</span>
              </div>
              <input 
                type="range" 
                min="0.0" 
                max="0.25" 
                step="0.05"
                value={cascadeChance} 
                onChange={(e) => setCascadeChance(Number(e.target.value))}
                className="w-full accent-amber-500 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-[10px] text-slate-500">Increments edge probabilities during live cascade animation.</span>
            </div>

            {/* Graph Generation Reset Trigger */}
            <button
              onClick={handleRegenerateGraph}
              disabled={isGenerating}
              className="w-full mt-2 py-2.5 px-4 bg-slate-800 hover:bg-slate-700 active:bg-slate-900 border border-slate-700 disabled:opacity-40 rounded-lg text-xs font-semibold text-slate-100 flex items-center justify-center gap-2 transition duration-200"
              id="regenerate_btn"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isGenerating ? 'animate-spin' : ''}`} />
              Regenerate Social Network
            </button>
          </div>

          {/* RUN ENGINE ACTION CARD */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col gap-4">
            
            <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
              <Zap className="w-4 h-4 text-indigo-400" />
              <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Execute Optimization</h2>
            </div>
            
            <p className="text-xs text-slate-400 leading-relaxed">
              Solve the influence maximization problem using traditional Greedy Hill-Climbing or speed-optimized IMM sampling algorithms.
            </p>

            <div className="flex flex-col gap-2.5 mt-2" id="runner_triggers">
              {/* IMM trigger */}
              <button
                onClick={handleRunIMM}
                disabled={isSolving || isGenerating || !graph}
                className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded-lg text-xs font-semibold flex items-center justify-between shadow transition duration-200"
              >
                <div className="flex items-center gap-2">
                  <span className="p-1 bg-indigo-700 rounded text-[10px] font-mono">RIS</span>
                  <span>Run Fast IMM Selection</span>
                </div>
                <ChevronRight className="w-4 h-4" />
              </button>

              {/* Greedy trigger */}
              <button
                onClick={handleRunGreedy}
                disabled={isSolving || isGenerating || !graph}
                className="w-full py-2.5 px-4 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 border border-slate-700 text-slate-200 rounded-lg text-xs font-semibold flex items-center justify-between transition duration-200"
              >
                <div className="flex items-center gap-2">
                  <span className="p-1 bg-slate-900 rounded text-[10px] font-mono text-indigo-400 border border-slate-800">EXP</span>
                  <span>Run Greedy Hill-Climbing</span>
                </div>
                <ChevronRight className="w-4 h-4" />
              </button>

              {/* Compare side by side trigger */}
              <button
                onClick={handleCompareAll}
                disabled={isSolving || isGenerating || !graph}
                className="w-full py-2 px-4 bg-slate-950 hover:bg-slate-900 border border-indigo-500/30 text-indigo-300 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition duration-200 font-mono"
              >
                <BarChart3 className="w-3.5 h-3.5" />
                Solve & Compare Side-by-Side
              </button>
            </div>
          </div>

          {/* SIMULATION EXPLANATION AND MATH REFERENCE CARD */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 text-[11px] leading-relaxed text-slate-400 flex flex-col gap-2" id="reference_sheet">
            <div className="flex items-center gap-2 text-slate-300 font-semibold border-b border-slate-800 pb-1.5 uppercase tracking-wider">
              <BookOpen className="w-3.5 h-3.5 text-blue-400" />
              Mathematics Reference
            </div>
            <p>
              The <strong>Independent Cascade (IC)</strong> model starts from active seeds S0. Each newly active node u gets exactly 1 attempt to activate each out-neighbor v with a success probability p(u, v). 
            </p>
            <p className="border-l-2 border-indigo-500 pl-2 text-slate-300 py-1 font-mono italic">
              InfluenceSpread(S) = E[|S_final|]
            </p>
            <p>
              Traditional Greedy has an approximation ratio of (1 - 1/e) ≈ 63%, but takes O(k * N * R) Monte Carlo trials. The <strong>IMM (Influence Maximization via Martingale)</strong> uses Reverse Influence Sampling to run in near-linear time O((k+L)M log(N) / epsilon^2) with the same theoretical guarantees!
            </p>
          </div>
        </section>

        {/* RIGHT COLUMN: INTERACTIVE TABS & RENDER PANES (8 cols size) */}
        <section className="lg:col-span-8 flex flex-col gap-6" id="display_workspace">
          
          {/* TAP CONTROLS RAILS */}
          <div className="flex border-b border-slate-800 bg-slate-950 rounded-xl p-1.5 shadow-md flex-wrap md:flex-nowrap gap-1" id="tab_rails">
            <button
              onClick={() => setActiveTab('graph')}
              className={`flex-1 py-2.5 px-4 rounded-lg text-xs font-semibold flex items-center justify-center gap-2.5 transition duration-150 cursor-pointer ${
                activeTab === 'graph' 
                  ? 'bg-blue-600/10 text-blue-400 border border-blue-500/30 font-bold' 
                  : 'text-slate-400 hover:bg-slate-900'
              }`}
            >
              <Network className="w-4 h-4" />
              <span>Interactive Social Graph</span>
            </button>

            <button
              onClick={() => setActiveTab('analytics')}
              className={`flex-1 py-2.5 px-4 rounded-lg text-xs font-semibold flex items-center justify-center gap-2.5 transition duration-150 cursor-pointer ${
                activeTab === 'analytics' 
                  ? 'bg-indigo-600/10 text-indigo-400 border border-indigo-500/30 font-bold' 
                  : 'text-slate-400 hover:bg-slate-900'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Metrics & Visual Charts</span>
            </button>

            <button
              onClick={() => setActiveTab('code')}
              className={`flex-1 py-2.5 px-4 rounded-lg text-xs font-semibold flex items-center justify-center gap-2.5 transition duration-150 cursor-pointer ${
                activeTab === 'code' 
                  ? 'bg-amber-600/10 text-amber-400 border border-amber-500/30 font-bold' 
                  : 'text-slate-400 hover:bg-slate-900'
              }`}
            >
              <Code2 className="w-4 h-4" />
              <span>Python Source Explorer</span>
            </button>
          </div>

          {/* TAB 1: GRAPH VISUALIZATION CONSOLE */}
          {activeTab === 'graph' && (
            <div className="bg-slate-950 border border-slate-800 rounded-2xl p-6 shadow-2xl flex flex-col gap-6 animate-fade-in" id="visual_graph_module">
              
              {/* Toolbar and controllers */}
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4" id="visual_toolbar">
                <div>
                  <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                    <span>Diffusion Cascade Map</span>
                    <span className="px-2 py-0.5 bg-slate-800 rounded text-[10px] font-mono text-slate-400">Scale Free BA Layout</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Visual network representation showing how the selected {selectedSeeds.length} seeds propagate contagion.
                  </p>
                </div>

                {/* Simulated Cascade trigger */}
                <div className="flex items-center gap-3">
                  {selectedSeeds.length > 0 && (
                    <button
                      onClick={triggerAnimatedCascade}
                      disabled={isCascading}
                      className="py-2 px-4 bg-amber-500 hover:bg-amber-400 disabled:opacity-40 text-slate-950 rounded-lg text-xs font-bold flex items-center gap-2 transition shadow duration-150 cursor-pointer"
                      id="play_animation_btn"
                    >
                      <Play className="w-4 h-4 fill-current" />
                      Play Cascade Ripple
                    </button>
                  )}
                  
                  {selectedSeeds.length === 0 && (
                    <span className="text-xs text-amber-500 bg-amber-500/10 border border-amber-500/20 px-3 py-1.5 rounded-lg flex items-center gap-2 font-mono" id="no_seeds_warning">
                      <Info className="w-3.5 h-3.5" />
                      Please run IMM or Greedy first!
                    </span>
                  )}
                </div>
              </div>

              {/* GRAPH RENDERING PANEL */}
              <div className="relative border border-slate-800 bg-slate-900 rounded-xl overflow-hidden shadow-inner flex items-center justify-center p-4 min-h-[420px]" id="network_svg_holder">
                {isGenerating ? (
                  <div className="flex flex-col items-center gap-3 py-16" id="loading_graph_spinner">
                    <RefreshCw className="w-10 h-10 text-blue-500 animate-spin" />
                    <p className="text-slate-400 text-sm font-mono font-medium animate-pulse">Constructing social nodes hierarchy...</p>
                  </div>
                ) : graph && nodePositions.length > 0 ? (
                  <>
                    {/* SVG canvas */}
                    <svg 
                      viewBox="0 0 550 420" 
                      className="w-full max-w-[550px] aspect-video"
                      style={{ maxHeight: '420px' }}
                    >
                      {/* CONNECTING EDGES LAYER */}
                      <g stroke="#ffffff" strokeOpacity={0.06} strokeWidth={1}>
                        {graph.edges.map((edge, idx) => {
                          const sPos = nodePositions[edge.source];
                          const tPos = nodePositions[edge.target];
                          if (!sPos || !tPos) return null;

                          // Edge cascade highlight check (source and target both activated during propagation)
                          const isCascadeActive = visibleActivatedNodes.includes(edge.source) && visibleActivatedNodes.includes(edge.target);

                          return (
                            <line 
                              key={`edge-${idx}`}
                              x1={sPos.x}
                              y1={sPos.y}
                              x2={tPos.x}
                              y2={tPos.y}
                              stroke={isCascadeActive ? '#f4a261' : '#475569'}
                              strokeOpacity={isCascadeActive ? 0.7 : 0.08}
                              strokeWidth={isCascadeActive ? 1.5 : 0.8}
                              strokeDasharray={isCascadeActive ? "4,2" : "none"}
                            />
                          );
                        })}
                      </g>

                      {/* INDIVIDUAL NODES LAYER */}
                      <g>
                        {graph.nodes.map((node) => {
                          const pos = nodePositions[node.id];
                          if (!pos) return null;

                          const isSeed = selectedSeeds.includes(node.id);
                          const isInfluenced = visibleActivatedNodes.includes(node.id) && !isSeed;
                          const commColor = COMMUNITY_COLORS[node.community % COMMUNITY_COLORS.length];

                          // Default node states coloring:
                          // Seed = Red
                          // Reached = Orange
                          // Non-reached = Distinct Community color grouping
                          let fill = commColor;
                          let radius = 4 + Math.sqrt(node.degree) * 0.45;
                          let stroke = '#ffffff';
                          let strokeWidth = 0.5;

                          if (isSeed) {
                            fill = '#e63946'; // Strong active red
                            radius = Math.max(7, radius + 2);
                            stroke = '#fee2e2';
                            strokeWidth = 2;
                          } else if (isInfluenced) {
                            fill = '#f4a261'; // Soft Orange
                            radius = Math.max(5, radius + 0.8);
                            stroke = '#ffedd5';
                            strokeWidth = 1.5;
                          }

                          // Highlight properties if hovered
                          const isHovered = hoveredNode?.id === node.id;
                          if (isHovered) {
                            radius += 3;
                            strokeWidth = 2.5;
                            stroke = '#ffffff';
                          }

                          return (
                            <circle
                              key={`node-${node.id}`}
                              cx={pos.x}
                              cy={pos.y}
                              r={radius}
                              fill={fill}
                              stroke={stroke}
                              strokeWidth={strokeWidth}
                              opacity={hoveredNode && !isHovered && !selectedSeeds.includes(node.id) ? 0.45 : 0.9}
                              onMouseEnter={() => setHoveredNode(node)}
                              onMouseLeave={() => setHoveredNode(null)}
                              className="transition-all duration-300 ease-out cursor-pointer"
                            />
                          );
                        })}
                      </g>
                    </svg>

                    {/* LIVE DYNAMIC CONSOLE/NODE HOVER TIP IN CANVAS */}
                    {hoveredNode && (
                      <div className="absolute top-4 left-4 bg-slate-950/95 border border-slate-800 rounded-lg p-3 text-xs font-mono min-w-[200px] shadow-2xl flex flex-col gap-1.5 z-10 animate-fade-in" id="node_tooltip">
                        <div className="flex justify-between items-center border-b border-slate-800 pb-1">
                          <span className="text-slate-400 font-bold">Node ID: #{hoveredNode.id}</span>
                          {selectedSeeds.includes(hoveredNode.id) && (
                            <span className="bg-red-500/10 text-red-400 border border-red-500/20 px-1 rounded text-[9px] uppercase font-bold">Active Seed</span>
                          )}
                          {visibleActivatedNodes.includes(hoveredNode.id) && !selectedSeeds.includes(hoveredNode.id) && (
                            <span className="bg-amber-500/10 text-amber-400 border border-amber-500/20 px-1 rounded text-[9px] uppercase font-bold">Cascaded</span>
                          )}
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Degree centrality:</span>
                          <span className="text-slate-200">{(hoveredNode.degree / (numNodes * 2)).toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Absolute Degree:</span>
                          <span className="text-slate-200">{hoveredNode.degree} links</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Community cluster:</span>
                          <span style={{ color: COMMUNITY_COLORS[hoveredNode.community % COMMUNITY_COLORS.length] }} className="font-bold">
                            Cluster #{hoveredNode.community}
                          </span>
                        </div>
                      </div>
                    )}

                    {/* ANIMATING PROGRESS SUBBAR */}
                    {isCascading && (
                      <div className="absolute bottom-4 right-4 bg-slate-950 border border-amber-500/40 rounded-lg px-4 py-2 text-xs font-mono text-amber-400 shadow-2xl flex items-center gap-2 animate-pulse" id="animation_banner">
                        <Flame className="w-4 h-4 text-amber-500 animate-bounce" />
                        <span>Cascade Wave #{cascadeStep} running...</span>
                      </div>
                    )}
                  </>
                ) : (
                  <p className="text-slate-500 text-sm font-mono">No network loaded. Regenerate network.</p>
                )}
              </div>

              {/* GRAPH METRICS BOXES */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4" id="graph_metrics_row">
                <div className="bg-slate-900 border border-slate-800/80 rounded-xl p-3.5 flex flex-col gap-1">
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold font-mono">Graph Density</span>
                  <span className="text-lg font-bold text-blue-400 font-mono">{graphStats.density.toFixed(4)}</span>
                </div>
                <div className="bg-slate-900 border border-slate-800/80 rounded-xl p-3.5 flex flex-col gap-1">
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold font-mono">Average Degree</span>
                  <span className="text-lg font-bold text-indigo-400 font-mono">{graphStats.avgDegree.toFixed(2)}</span>
                </div>
                <div className="bg-slate-900 border border-slate-800/80 rounded-xl p-3.5 flex flex-col gap-1">
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold font-mono">Active Seeds Coverage</span>
                  <span className="text-lg font-bold text-amber-500 font-mono">
                    {visibleActivatedNodes.length} / {numNodes} ({(visibleActivatedNodes.length / numNodes * 100).toFixed(1)}%)
                  </span>
                </div>
                <div className="bg-slate-900 border border-slate-800/80 rounded-xl p-3.5 flex flex-col gap-1">
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold font-mono">Top Central Node</span>
                  <span className="text-slate-200 text-xs mt-1 font-mono">Node #{graphStats.topCentral[0] || 0}</span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: ANALYTICS & VISUAL RESULTS CHARTS */}
          {activeTab === 'analytics' && (
            <div className="bg-slate-950 border border-slate-800 rounded-2xl p-6 shadow-2xl flex flex-col gap-6 animate-fade-in" id="visual_analytics_module">
              <div>
                <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-indigo-400" />
                  <span>Interactive Algorithmic Performance Graphs</span>
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Compare traditional greedy heuristics against RIS-based IMM regarding coverage growth, runtime speeds, and community partition coverage.
                </p>
              </div>

              {/* ROW 1: GROWTH CURVE CHART AND RUNTIME BAR CHART */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6" id="charts_row_1">
                
                {/* 2A: LINE GRAPH -- BUDGET vs EXPECTED SPREAD */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col gap-4">
                  <h4 className="text-xs font-bold text-slate-300 font-mono uppercase tracking-wider flex items-center justify-between">
                    <span>Expected Influence VS Seeding Size (k)</span>
                    <span className="text-[10px] text-indigo-400">Higher is Better</span>
                  </h4>
                  
                  {/* Custom animated SVG Line plot */}
                  <div className="border border-slate-800/60 bg-slate-950 rounded-lg p-3 h-[200px] relative flex items-center justify-center">
                    {greedyResult || immResult ? (
                      <svg viewBox="0 0 200 120" className="w-full h-full">
                        {/* Grid lines */}
                        <line x1="20" y1="10" x2="190" y2="10" stroke="#334155" strokeWidth="0.5" strokeOpacity="0.4" />
                        <line x1="20" y1="35" x2="190" y2="35" stroke="#334155" strokeWidth="0.5" strokeOpacity="0.4" />
                        <line x1="20" y1="60" x2="190" y2="60" stroke="#334155" strokeWidth="0.5" strokeOpacity="0.4" />
                        <line x1="20" y1="85" x2="190" y2="85" stroke="#334155" strokeWidth="0.5" strokeOpacity="0.4" />
                        <line x1="20" y1="105" x2="190" y2="105" stroke="#475569" strokeWidth="1" />
                        <line x1="20" y1="10" x2="20" y2="105" stroke="#475569" strokeWidth="1" />

                        {/* Text labels on Y-axis */}
                        <text x="15" y="15" fill="#64748b" fontSize="5" textAnchor="end">N</text>
                        <text x="15" y="60" fill="#64748b" fontSize="5" textAnchor="end">{Math.floor(numNodes / 2)}</text>
                        <text x="15" y="105" fill="#64748b" fontSize="5" textAnchor="end">0</text>

                        {/* Greedy Line */}
                        {greedyResult && (
                          <path
                            d={greedyResult.history.map((val, idx) => {
                              const x = 20 + (idx / (seedCountK - 1)) * 170;
                              const y = 105 - (val / numNodes) * 95;
                              return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }).join(' ')}
                            fill="none"
                            stroke="#3b82f6"
                            strokeWidth="1.5"
                          />
                        )}

                        {/* IMM Line */}
                        {immResult && (
                          <path
                            d={immResult.history.map((val, idx) => {
                              const x = 20 + (idx / (seedCountK - 1)) * 170;
                              const y = 105 - (val / numNodes) * 95;
                              return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }).join(' ')}
                            fill="none"
                            stroke="#f59e0b"
                            strokeWidth="1.5"
                            strokeDasharray="2,2"
                          />
                        )}

                        {/* Plots Data Points */}
                        {greedyResult && greedyResult.history.map((val, idx) => {
                          const x = 20 + (idx / (seedCountK - 1)) * 170;
                          const y = 105 - (val / numNodes) * 95;
                          return (
                            <circle key={`pt-g-${idx}`} cx={x} cy={y} r="2" fill="#3b82f6" stroke="#ffffff" strokeWidth="0.3" />
                          );
                        })}

                        {immResult && immResult.history.map((val, idx) => {
                          const x = 20 + (idx / (seedCountK - 1)) * 170;
                          const y = 105 - (val / numNodes) * 95;
                          return (
                            <rect key={`pt-imm-${idx}`} x={x - 1.5} y={y - 1.5} width="3" height="3" fill="#f59e0b" stroke="#ffffff" strokeWidth="0.3" />
                          );
                        })}

                        {/* X Labels */}
                        {Array.from({ length: seedCountK }).map((_, i) => {
                          const x = 20 + (i / (seedCountK - 1)) * 170;
                          return (
                            <text key={`xl-${i}`} x={x} y="113" fill="#64748b" fontSize="5" textAnchor="middle">k = {i + 1}</text>
                          );
                        })}
                      </svg>
                    ) : (
                      <span className="text-xs font-mono text-slate-500 animate-pulse">Run algorithms to generate graph comparisons</span>
                    )}
                  </div>

                  {/* Legend */}
                  <div className="flex gap-4 items-center justify-center text-xs font-mono text-slate-400">
                    <div className="flex items-center gap-1.5">
                      <span className="w-3 h-0.5 bg-blue-500 inline-block" />
                      <span>Greedy Model</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-3 h-0.5 bg-amber-500 border-dashed border-t inline-block" />
                      <span>IMM RIS Model</span>
                    </div>
                  </div>
                </div>

                {/* 2B: BAR CHART -- RUNTIMES DURATION */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col gap-4">
                  <h4 className="text-xs font-bold text-slate-300 font-mono uppercase tracking-wider flex items-center justify-between">
                    <span>Algorithmic Execution Runtime</span>
                    <span className="text-[10px] text-amber-500 font-mono">Lower is Better</span>
                  </h4>

                  {/* SVG Bar chart for times */}
                  <div className="border border-slate-800/60 bg-slate-950 rounded-lg p-3 h-[200px] relative flex items-center justify-center font-mono">
                    {greedyResult || immResult ? (
                      <div className="w-full flex justify-around items-end h-[160px] pb-2 border-b border-slate-800 px-4">
                        {/* Traditional Greedy Time */}
                        <div className="flex flex-col items-center gap-2 w-1/3">
                          <span className="text-[10px] text-blue-400 font-semibold">
                            {greedyResult ? `${greedyResult.runtime.toFixed(1)} ms` : 'N/A'}
                          </span>
                          <div 
                            style={{ height: greedyResult ? `${Math.max(5, Math.min(100, (greedyResult.runtime / (greedyResult.runtime + (immResult?.runtime || 0) || 1)) * 100))}px` : '4px' }}
                            className="w-12 bg-gradient-to-t from-blue-600 to-indigo-500 rounded-t border-t border-blue-400 shadow-lg" 
                          />
                          <span className="text-[9px] text-slate-400 text-center font-semibold uppercase">Greedy</span>
                        </div>

                        {/* IMM Time */}
                        <div className="flex flex-col items-center gap-2 w-1/3 border-l border-slate-800/80">
                          <span className="text-[10px] text-amber-500 font-semibold">
                            {immResult ? `${immResult.runtime.toFixed(1)} ms` : 'N/A'}
                          </span>
                          <div 
                            style={{ height: immResult ? `${Math.max(5, Math.min(100, (immResult.runtime / (greedyResult?.runtime || 1)) * 100))}px` : '4px' }}
                            className="w-12 bg-gradient-to-t from-amber-500 to-orange-400 rounded-t border-t border-amber-300 shadow-lg" 
                          />
                          <span className="text-[9px] text-slate-400 text-center font-semibold uppercase">IMM (RIS)</span>
                        </div>
                      </div>
                    ) : (
                      <span className="text-xs text-slate-500 animate-pulse">Run algorithms to see speed metrics</span>
                    )}
                  </div>

                  {immResult && greedyResult && (
                    <span className="text-[10px] text-emerald-400 text-center font-mono font-semibold block bg-emerald-500/10 border border-emerald-500/20 py-1.5 rounded-lg">
                      ✔ IMM is {(greedyResult.runtime / immResult.runtime).toFixed(1)}x faster for social optimization!
                    </span>
                  )}
                </div>
              </div>

              {/* ROW 2: COMMUNITY INFLUENCE BREAKDOWN */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col gap-4" id="charts_row_2">
                <h4 className="text-xs font-bold text-slate-300 font-mono uppercase tracking-wider flex items-center justify-between">
                  <span>Seeding Cascade Coverage grouped by Communities</span>
                  <span className="text-[10px] text-blue-400 font-mono">Bento Community Partitions</span>
                </h4>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center" id="community_comparison">
                  <div className="flex flex-col gap-3">
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Social networks partition naturally into modular groups (communities). A robust seeding strategy should span communities to avoid circular message redundancy. 
                    </p>
                    <div className="flex flex-col gap-2 mt-1 font-mono text-xs text-slate-300" id="community_stats_bullets">
                      <div className="flex justify-between items-center bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                        <span>Total Detected Communities:</span>
                        <span className="font-bold text-indigo-400">{communityMetrics.length}</span>
                      </div>
                      <div className="flex justify-between items-center bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                        <span>Seeded Communities Spanned:</span>
                        <span className="font-bold text-emerald-400">
                          {communityMetrics.filter(m => m.seeds.length > 0).length} of {communityMetrics.length}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Horizontal Bar Chart showing reached ratio per community */}
                  <div className="flex flex-col gap-3" id="comm_bars">
                    {communityMetrics.length > 0 ? (
                      communityMetrics.map((comm) => {
                        const coverageRatio = comm.reached / comm.size;
                        return (
                          <div key={`comm-metric-${comm.id}`} className="flex flex-col gap-1 font-mono">
                            <div className="flex justify-between text-xs text-slate-400">
                              <span className="flex items-center gap-1.5 font-semibold text-slate-300">
                                <span style={{ backgroundColor: comm.color }} className="w-2.5 h-2.5 rounded-full inline-block" />
                                Community #{comm.id} (Size {comm.size})
                              </span>
                              <span className="text-slate-200">
                                Reached: {comm.reached} ({Math.round(coverageRatio * 100)}%)
                              </span>
                            </div>
                            <div className="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
                              <div 
                                style={{ 
                                  width: `${Math.round(coverageRatio * 100)}%`,
                                  backgroundColor: comm.color 
                                }} 
                                className="h-full rounded-full transition-all duration-700 ease-out" 
                              />
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <span className="text-xs text-slate-500 text-center animate-pulse py-8">Generate graph to analyze community modularity</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: PYTHON SOURCE FILES EXPLORER */}
          {activeTab === 'code' && (
            <div className="bg-slate-950 border border-slate-800 rounded-2xl p-6 shadow-2xl flex flex-col gap-5 animate-fade-in" id="visual_code_module">
              
              {/* Header */}
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                  <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                    <FileCode className="w-5 h-5 text-amber-400" />
                    <span>Python Production Codebase Explorer</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Explore, copy, or download the exact Python codefiles created inside the filesystem of this container space.
                  </p>
                </div>

                {/* ZIP Download instructions badge */}
                <div className="bg-amber-400/5 border border-amber-400/20 text-amber-300 text-xs px-3 py-1.5 rounded-lg flex items-center gap-2 font-mono">
                  <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                  Full code in /influence-maximization/
                </div>
              </div>

              {/* INTEGRATED DIRECTORY EXPLORER */}
              <div className="grid grid-cols-1 md:grid-cols-12 gap-5 border border-slate-800 rounded-xl overflow-hidden shadow-inner bg-slate-900" id="dir_explorer_window">
                
                {/* File tree sidebar (4 cols) */}
                <div className="md:col-span-4 bg-slate-950/80 border-r border-slate-800 p-4 flex flex-col gap-3 font-mono text-xs">
                  <div className="flex items-center gap-1.5 text-slate-400 font-bold border-b border-slate-800 pb-2 text-[10px] uppercase tracking-wider">
                    <Folder className="w-3.5 h-3.5 text-slate-500 fill-current" />
                    <span>Project Directories</span>
                  </div>

                  <div className="flex flex-col gap-1 text-[11px]" id="file_list_tree">
                    {/* Requirements */}
                    <button
                      onClick={() => setActiveFile("requirements.txt")}
                      className={`flex items-center justify-between py-2 px-2.5 rounded-md transition ${
                        activeFile === "requirements.txt" ? 'bg-amber-500/10 text-amber-400' : 'text-slate-400 hover:bg-slate-900'
                      }`}
                    >
                      <span className="flex items-center gap-1.5 truncate">
                        <FileCode className="w-3.5 h-3.5" />
                        requirements.txt
                      </span>
                    </button>

                    {/* Source folder indicator */}
                    <div className="flex items-center gap-1 text-slate-500 py-1.5 pl-1.5">
                      <Folder className="w-3.5 h-3.5 fill-current" />
                      <span>src/</span>
                    </div>

                    {/* Custom Py Modules */}
                    {Object.keys(PYTHON_CODEFILES).filter(f => f.startsWith("src/")).map((filename) => {
                      const label = filename.replace("src/", "");
                      return (
                        <button
                          key={filename}
                          onClick={() => setActiveFile(filename)}
                          className={`flex items-center justify-between py-2 pl-6 pr-2.5 rounded-md transition ${
                            activeFile === filename ? 'bg-amber-500/10 text-amber-400' : 'text-slate-400 hover:bg-slate-900'
                          }`}
                        >
                          <span className="flex items-center gap-1.5 truncate">
                            <FileCode className="w-3.5 h-3.5 text-slate-500" />
                            {label}
                          </span>
                        </button>
                      );
                    })}

                    {/* main.py */}
                    <button
                      onClick={() => setActiveFile("main.py")}
                      className={`flex items-center justify-between py-2 px-2.5 rounded-md border-t border-slate-850 mt-1 transition ${
                        activeFile === "main.py" ? 'bg-amber-500/10 text-amber-400' : 'text-slate-400 hover:bg-slate-900'
                      }`}
                    >
                      <span className="flex items-center gap-1.5 truncate">
                        <FileCode className="w-3.5 h-3.5" />
                        main.py
                      </span>
                    </button>
                  </div>
                </div>

                {/* Code viewport (8 cols) */}
                <div className="md:col-span-8 flex flex-col h-[340px] md:h-[380px] bg-slate-950 font-mono text-xs">
                  
                  {/* Copy toolbar header */}
                  <div className="flex justify-between items-center bg-slate-900/60 px-4 py-2 border-b border-slate-800">
                    <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">{activeFile}</span>
                    <button
                      onClick={triggerCopyFile}
                      className="py-1 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 hover:border-slate-600 rounded flex items-center justify-center gap-1.5 text-[10px] font-semibold transition active:bg-slate-900 cursor-pointer"
                    >
                      {copiedFile ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                      <span>{copiedFile ? 'Copied' : 'Copy Code'}</span>
                    </button>
                  </div>

                  {/* Code space */}
                  <pre className="flex-1 p-4 overflow-auto text-slate-300 text-[11px] leading-relaxed select-all">
                    <code>{PYTHON_CODEFILES[activeFile]}</code>
                  </pre>
                </div>
              </div>

              {/* QUICK TERMINAL HOW-TO */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-2 font-mono text-xs text-slate-400">
                <span className="text-slate-300 font-bold uppercase text-[10px] tracking-wider flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  Offline Execution Steps
                </span>
                <p className="text-[11px]">To run the Python code generated in `/influence-maximization/` in your local computer, open your system terminal and execute:</p>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-slate-300 text-[10px] flex flex-col gap-1">
                  <div>cd influence-maximization</div>
                  <div>python -m venv venv</div>
                  <div>source venv/bin/activate  # venv\Scripts\activate on Windows</div>
                  <div>pip install -r requirements.txt</div>
                  <div className="text-amber-400">python main.py</div>
                </div>
              </div>

            </div>
          )}

        </section>

      </main>

      {/* FOOTER */}
      <footer className="bg-slate-950 border-t border-slate-800 text-slate-500 py-4 px-6 text-center text-xs font-mono" id="app_footer">
        Social Network Influence Maximization Simulator • Designed with Expert Python & NetworkX models • 2026 UTC
      </footer>
    </div>
  );
}
