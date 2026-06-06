import { NetworkGraph, NetworkNode, NetworkEdge, SimulationResult } from './types';

/**
 * Generates a Scale-Free Barabasi-Albert social network graph
 * and converts it to bidirectional edges with random propagation probabilities.
 */
export function buildSyntheticGraphJS(n: number, m: number): NetworkGraph {
  const nodes: NetworkNode[] = [];
  const edges: NetworkEdge[] = [];
  const adjacency = new Map<number, number[]>();
  const inAdjacency = new Map<number, number[]>();

  // Initialize adjacency maps
  for (let i = 0; i < n; i++) {
    adjacency.set(i, []);
    inAdjacency.set(i, []);
  }

  // Pre-fill active node registry for preferential attachment algorithm
  const repeatedNodes: number[] = [];

  // 1. Core starter clique (first m nodes are connected to each other)
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < m; j++) {
      // Directed edge i -> j
      const prob1 = 0.01 + Math.random() * 0.09;
      edges.push({ source: i, target: j, prob: prob1 });
      adjacency.get(i)!.push(j);
      inAdjacency.get(j)!.push(i);

      // Directed edge j -> i
      const prob2 = 0.01 + Math.random() * 0.09;
      edges.push({ source: j, target: i, prob: prob2 });
      adjacency.get(j)!.push(i);
      inAdjacency.get(i)!.push(j);

      repeatedNodes.push(i, j);
    }
  }

  // 2. Grow the graph using preferential attachment
  for (let sourceNode = m; sourceNode < n; sourceNode++) {
    const targets = new Set<number>();
    
    // Choose m distinct targets proportionally to degree
    let attempts = 0;
    while (targets.size < m && attempts < 100) {
      const idx = Math.floor(Math.random() * repeatedNodes.length);
      const prospectiveTarget = repeatedNodes[idx];
      if (prospectiveTarget !== sourceNode) {
        targets.add(prospectiveTarget);
      }
      attempts++;
    }

    // Connect node to preferred targets bidirectionally
    targets.forEach((targetNode) => {
      // forward edge
      const p1 = 0.01 + Math.random() * 0.09;
      edges.push({ source: sourceNode, target: targetNode, prob: p1 });
      adjacency.get(sourceNode)!.push(targetNode);
      inAdjacency.get(targetNode)!.push(sourceNode);

      // backward edge
      const p2 = 0.01 + Math.random() * 0.09;
      edges.push({ source: targetNode, target: sourceNode, prob: p2 });
      adjacency.get(targetNode)!.push(sourceNode);
      inAdjacency.get(sourceNode)!.push(targetNode);

      repeatedNodes.push(sourceNode, targetNode);
    });
  }

  // Compile final node statistics
  for (let i = 0; i < n; i++) {
    const outDegree = adjacency.get(i)!.length;
    const inDegree = inAdjacency.get(i)!.length;
    nodes.push({
      id: i,
      degree: outDegree + inDegree,
      community: 0, // Assigned during community detection
    });
  }

  const graph = { nodes, edges, adjacency, inAdjacency };

  // Detect communities using Label Propagation (LPA)
  detectCommunitiesLPA(graph);

  return graph;
}

/**
 * Fast Label Propagation Algorithm (LPA) for Community Detection.
 * Assigns a community index to each node in the network.
 */
function detectCommunitiesLPA(graph: NetworkGraph): void {
  const n = graph.nodes.length;
  const labels: number[] = Array.from({ length: n }, (_, i) => i);

  // Run 5 iterations of Label Propagation
  const iterations = 5;
  for (let iter = 0; iter < iterations; iter++) {
    // Traverse nodes in random order to prevent bias
    const order = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - 0.5);

    for (const node of order) {
      const neighbors = [
        ...(graph.adjacency.get(node) || []),
        ...(graph.inAdjacency.get(node) || []),
      ];

      if (neighbors.length === 0) continue;

      // Count label frequencies among neighbors
      const counts = new Map<number, number>();
      for (const neighbor of neighbors) {
        const lbl = labels[neighbor];
        counts.set(lbl, (counts.get(lbl) || 0) + 1);
      }

      // Pick the label with highest frequency
      let bestLabel = labels[node];
      let maxCount = -1;
      counts.forEach((cnt, lbl) => {
        if (cnt > maxCount) {
          maxCount = cnt;
          bestLabel = lbl;
        }
      });

      labels[node] = bestLabel;
    }
  }

  // Compress labels to unique, contiguous community indices starting from 0
  const uniqueLabels = Array.from(new Set(labels));
  const labelToId = new Map<number, number>();
  uniqueLabels.forEach((lbl, index) => {
    labelToId.set(lbl, index);
  });

  graph.nodes.forEach((node) => {
    node.community = labelToId.get(labels[node.id]) || 0;
  });
}

/**
 * Runs the Independent Cascade (IC) simulation from a set of starting seeds,
 * returning the expected coverage score and storing one single sample simulation
 * active nodes array for rendering animations in the UI.
 */
export function simulateICJS(
  graph: NetworkGraph,
  seeds: number[],
  numSimulations: number
): { expectedCoverage: number; singleSampleCascade: number[] } {
  if (seeds.length === 0) {
    return { expectedCoverage: 0, singleSampleCascade: [] };
  }

  let totalCoverage = 0;
  let representativeCascade: number[] = [];

  // Pre-compile edge map for fast hash checks
  const edgeProbMap = new Map<string, number>();
  graph.edges.forEach((edge) => {
    edgeProbMap.set(`${edge.source}_${edge.target}`, edge.prob);
  });

  for (let sim = 0; sim < numSimulations; sim++) {
    const active = new Set<number>(seeds);
    const queue: number[] = [...seeds];

    while (queue.length > 0) {
      const u = queue.shift()!;
      const neighbors = graph.adjacency.get(u) || [];

      for (const v of neighbors) {
        if (!active.has(v)) {
          const prob = edgeProbMap.get(`${u}_${v}`) || 0.0;
          if (Math.random() < prob) {
            active.add(v);
            queue.push(v);
          }
        }
      }
    }

    totalCoverage += active.size;

    // Save the first simulation as our representative visual case
    if (sim === 0) {
      representativeCascade = Array.from(active);
    }
  }

  return {
    expectedCoverage: totalCoverage / numSimulations,
    singleSampleCascade: representativeCascade,
  };
}

/**
 * Runs Greedy influence maximization algorithm in TypeScript.
 * We use an optimized number of simulations to guarantee instant UI responsiveness.
 */
export function runGreedyJS(
  graph: NetworkGraph,
  k: number,
  numSimulations: number,
  onStepProgress?: (round: number, bestNode: number, currentCoverage: number) => void
): SimulationResult {
  const startTime = performance.now();
  const seeds: number[] = [];
  const history: number[] = [];

  for (let step = 1; step <= k; step++) {
    let bestCandidate = -1;
    let bestCoverage = -1;

    for (const node of graph.nodes) {
      const nodeId = node.id;
      if (seeds.includes(nodeId)) continue;

      // Temporary candidate seed set
      const candidateSeeds = [...seeds, nodeId];
      const { expectedCoverage } = simulateICJS(graph, candidateSeeds, numSimulations);

      if (expectedCoverage > bestCoverage) {
        bestCoverage = expectedCoverage;
        bestCandidate = nodeId;
      }
    }

    if (bestCandidate !== -1) {
      seeds.push(bestCandidate);
      history.push(bestCoverage);
      if (onStepProgress) {
        onStepProgress(step, bestCandidate, bestCoverage);
      }
    }
  }

  const duration = performance.now() - startTime;
  return {
    seeds,
    history,
    runtime: duration,
  };
}

/**
 * Helper to generate a single Reverse Reachable (RR) set from a target node.
 */
export function generateRRSetJS(graph: NetworkGraph, startNode: number, edgeProbMap: Map<string, number>): Set<number> {
  const visited = new Set<number>([startNode]);
  const queue: number[] = [startNode];

  while (queue.length > 0) {
    const curr = queue.shift()!;
    const incomingNeighbors = graph.inAdjacency.get(curr) || [];

    for (const pre of incomingNeighbors) {
      if (!visited.has(pre)) {
        // Prob of edge pre -> curr
        const prob = edgeProbMap.get(`${pre}_${curr}`) || 0.0;
        if (Math.random() < prob) {
          visited.add(pre);
          queue.push(pre);
        }
      }
    }
  }

  return visited;
}

/**
 * Runs IMM/RIS-based influence maximization algorithm in TypeScript.
 * Extremely high speed execution.
 */
export function runIMMJS(
  graph: NetworkGraph,
  k: number,
  numRRSets: number
): SimulationResult {
  const startTime = performance.now();
  const numNodes = graph.nodes.length;
  
  if (numNodes === 0 || k === 0) {
    return { seeds: [], history: [], runtime: 0 };
  }

  // Pre-compile edge probabilities
  const edgeProbMap = new Map<string, number>();
  graph.edges.forEach((edge) => {
    edgeProbMap.set(`${edge.source}_${edge.target}`, edge.prob);
  });

  // Step 1: Generate random RR sets
  const rrSets: Set<number>[] = [];
  for (let i = 0; i < numRRSets; i++) {
    const targetNode = Math.floor(Math.random() * numNodes);
    rrSets.push(generateRRSetJS(graph, targetNode, edgeProbMap));
  }

  // Step 2: Map each node to the indexing rows of the RR sets containing it
  const nodeToRR = new Map<number, Set<number>>();
  graph.nodes.forEach((node) => {
    nodeToRR.set(node.id, new Set<number>());
  });

  rrSets.forEach((rr, rrIdx) => {
    rr.forEach((nodeId) => {
      nodeToRR.get(nodeId)?.add(rrIdx);
    });
  });

  // Step 3: Greedy choice over RR sets
  const seeds: number[] = [];
  const coveredRRSetIndices = new Set<number>();
  const history: number[] = [];

  for (let step = 1; step <= k; step++) {
    let bestNode = -1;
    let maxCoverage = -1;

    for (const node of graph.nodes) {
      const nodeId = node.id;
      if (seeds.includes(nodeId)) continue;

      // Count uncovered RR sets containing this node
      let uncoveredCount = 0;
      const rrList = nodeToRR.get(nodeId);
      if (rrList) {
        rrList.forEach((idx) => {
          if (!coveredRRSetIndices.has(idx)) {
            uncoveredCount++;
          }
        });
      }

      if (uncoveredCount > maxCoverage) {
        maxCoverage = uncoveredCount;
        bestNode = nodeId;
      }
    }

    if (bestNode === -1) {
      break;
    }

    seeds.push(bestNode);
    
    // Union newly covered RR sets
    const matchRRs = nodeToRR.get(bestNode);
    if (matchRRs) {
      matchRRs.forEach((idx) => {
        coveredRRSetIndices.add(idx);
      });
    }

    // RIS score estimator: Fraction covered * absolute graph size
    const estimatedCoverage = (coveredRRSetIndices.size / numRRSets) * numNodes;
    history.push(estimatedCoverage);
  }

  const duration = performance.now() - startTime;
  return {
    seeds,
    history,
    runtime: duration,
  };
}
