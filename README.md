# RL & Path Planning Algorithms

A collection of reinforcement learning and path planning algorithm implementations, organized by domain.

## Repository Structure

### `cartpole/`

Implementations of RL algorithms applied to the CartPole-v1 Gymnasium environment. Each algorithm is self-contained and follows a consistent training/evaluation pattern.

**Implemented algorithms:**
- Proximal Policy Optimization (PPO) — policy gradient method with clipped surrogate objective
- Tabular Q-Learning — off-policy TD control using a discretized state space
- Tabular SARSA — on-policy TD control using a discretized state space
- Deep Q-Network (DQN) — off-policy TD control with a neural network function approximator and experience replay

**Planned:**
- A2C (Advantage Actor-Critic) — synchronous policy gradient with a shared value baseline
- SAC (Soft Actor-Critic) — off-policy maximum-entropy actor-critic for continuous actions
- TD3 (Twin Delayed DDPG) — deterministic policy gradient with double critics and delayed updates
- Rainbow DQN — DQN with prioritized replay, dueling networks, and multi-step returns

See [`cartpole/README.md`](./cartpole/README.md) for setup and training commands.

### `pathplanning-alg/`

Implementations of classical path planning algorithms for navigation in continuous and grid-based environments.

**Implemented algorithms:**
- RRT* (Optimal Rapidly-exploring Random Tree) — sampling-based planner with asymptotic optimality via rewiring
- A* — heuristic graph search for shortest path on discrete grids
- Dijkstra — uniform-cost graph search
- Bezier curves — smooth trajectory generation via parametric polynomial curves

**Planned:**
- RRT — baseline sampling-based planner without rewiring
- PRM (Probabilistic Roadmap Method) — sampling-based planner for multi-query environments
- D* Lite — incremental heuristic search for dynamic replanning
- Potential Fields — gradient-based reactive planner using attractive/repulsive force fields
- Hybrid A* — kinematically feasible A* variant for non-holonomic vehicles

## Setup

Each subdirectory is independently structured with its own dependencies. Navigate into the relevant directory and follow the local README for environment setup and run instructions.
