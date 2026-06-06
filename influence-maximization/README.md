# Influence Maximization in Social Networks

An engineered, high-performance, and mathematically grounded Social Network analysis tool containing implementations of traditional, high-quality heuristic greedy structures alongside advanced, modern Reverse Influence Sampling (RIS/IMM) algorithms.

---

## What is Influence Maximization?
Influence Maximization is the problem of identifying a small, elite set of "seed" nodes (influencers) in a social network that can trigger the largest possible cascade of information, viral adoption, or behavior adoption. By modeling propagation dynamics as a stochastic process, we analyze network topologies to optimize spread reach across social networks, communities, and digital clusters.

---

## Architecture Flow

```
                      +-----------------------------+
                      |     Social Network Graph    |
                      |  (Synthetic BA or Custom)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Independent Cascade Model  |
                      |   (Stochastic BFS Conduit)  |
                      +--------------+--------------+
                                     |
             +-----------------------+----------------------+
             |                                              |
             v                                              v
+--------------------------+                  +---------------------------+
|  Greedy Seeding Solvers  |                  |    IMM Algorithm Space    |
| (Baseline Hill Climbing) |                  | (Reverse Reachable Sets)  |
+------------+-------------+                  +-------------+-------------+
             |                                              |
             +-----------------------+----------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Results Report & Analytics |
                      |    (Save Comparison Plots)  |
                      +-----------------------------+
```

---

## How to Install and Run Locally

To execute this Python project locally, clone or copy the project files to your desktop and follow these terminal commands:

1. **Establish a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. **Install all locked dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Orchestrate tests and generate evaluation graphics:**
   ```bash
   python main.py
   ```

---

## Results & Analytics

After running the orchestrator, you can locate four generated diagnostic metrics plots within the `results/` directory path:

### 1. Cascade Spread Network Rendering (`results/influence_spread_graph.png`)
*Insert plot images here* - Visual layout of active seed sets, activated propagation paths, and non-reached nodes in a core subnetwork cluster.

### 2. Seeding Growth Curve Comparison (`results/greedy_vs_imm_coverage.png`)
*Insert plot images here* - Line growth metric displaying expected activated network size matching each size limit `k` under Greedy vs IMM.

### 3. Execution Performance Durations (`results/runtime_comparison.png`)
*Insert plot images here* - Comparative execution times for Traditional Greedy versus High-Performance IMM.

### 4. Cascade Coverage Breakdown by Communities (`results/community_influence_breakdown.png`)
*Insert plot images here* - Modular breakdown showing average reached node counts grouped inside unique community partitions.

---

## Real World Applications

- **Viral Marketing & Strategic Outreach:** Maximizing consumer reach and brand impressions given capped ambassador marketing budgets.
- **Epidemiology & Public Health:** Selecting high-contact social hubs to target for early vaccination, medical interventions, or viral awareness updates to stop infectious spreads.
- **Cybersecurity defense measures:** Strategic deployment of diagnostic firewalls, honeypots, or node-recovery processes on network hubs to isolate cascading malware infections.
