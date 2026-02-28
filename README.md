# RL & Path Planning Algorithms

A collection of reinforcement learning and path planning algorithm implementations, organized by domain.

## Repository Structure

### `cartpole/`

Implementations of RL algorithms applied to the CartPole-v1 Gymnasium environment. Each algorithm is self-contained and follows a consistent training/evaluation pattern.

**Implemented algorithms:**
- Proximal Policy Optimization (PPO) — policy gradient method with clipped surrogate objective
- Tabular Q-Learning — off-policy TD control using a discretized state space
- Tabular SARSA — on-policy TD control using a discretized state space

**Planned:**
- DQN, A2C, SAC

See [`cartpole/README.md`](./cartpole/README.md) for setup and training commands.

### `pathplanning-alg/`

Implementations of classical path planning algorithms for navigation in continuous and grid-based environments.

**Implemented algorithms:**
- RRT* (Optimal Rapidly-exploring Random Tree) — sampling-based planner with asymptotic optimality via rewiring

**Planned:**
- A* — heuristic graph search for shortest path on discrete grids
- RRT — baseline sampling-based planner without rewiring
- Dijkstra — uniform-cost graph search

## Setup

Each subdirectory is independently structured with its own dependencies. Navigate into the relevant directory and follow the local README for environment setup and run instructions.
