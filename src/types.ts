/**
 * Core Types for the Social Network Influence Maximization Simulator
 */

export interface NetworkEdge {
  source: number;
  target: number;
  prob: number; // Propagation probability
}

export interface NetworkNode {
  id: number;
  degree: number;
  community: number;
}

export interface NetworkGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  adjacency: Map<number, number[]>; // node -> outgoing neighbor indices
  inAdjacency: Map<number, number[]>; // node -> incoming neighbor indices
}

export interface SimulationResult {
  seeds: number[];
  history: number[]; // Estimated reach after picking each seed
  runtime: number; // Execution duration in ms
}

export interface CommunityMetric {
  id: number;
  size: number;
  color: string;
  seeds: number[];
  reached: number;
}
