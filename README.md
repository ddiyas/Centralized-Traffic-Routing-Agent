# Centralized Traffic Routing Agent

A centralized traffic routing system that uses PPO (Proximal Policy Optimization) to avoid Braess' Paradox inefficiencies and optimize system-wide traffic flow.

## Overview

This project implements a reinforcement learning solution to the classic Braess' Paradox problem in traffic routing. Instead of allowing selfish routing where individual drivers choose their own paths, a centralized agent makes routing decisions that optimize overall system performance.

**Key Result**: Achieves 33% improvement in average travel time compared to selfish routing scenarios.

## Background: Braess' Paradox

Braess' Paradox demonstrates that adding a new road to a traffic network can actually increase overall travel time when drivers route selfishly. The theoretical "price of anarchy" for this scenario is 4/3, meaning selfish routing can be 33% worse than optimal centralized routing.

## Technical Approach

### Environment (`traffic_env.py`)
- **Custom Gymnasium Environment**: 4-node traffic network (A→B→D, A→C→D with congestion-prone direct routes)
- **State Space**: Edge weights, active car counts, and car positions/status
- **Action Space**: Multi-discrete actions for next-node selection at decision points
- **Reward Function**: Negative system-wide travel time (encourages faster overall completion)

### Network Model
- **Congestion Model**: Travel time = base_time + max(cars - capacity, 0) × jam_factor
- **Dynamic Routing**: Cars can be rerouted at intermediate nodes based on agent decisions
- **Real-time Updates**: Edge weights updated each timestep based on current traffic load

### Training (`train_agent.py`)
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Scale**: 400 cars per episode
- **Parallel Training**: 4 environments for sample efficiency
- **Evaluation**: Multi-episode testing with arrival rate and completion time metrics

## Results

- **33% Travel Time Improvement** over selfish routing baseline
- **System-wide Optimization**: Agent learns to balance load across available routes
- **Braess' Paradox Avoidance**: Centralized control prevents the inefficiencies of selfish routing

## Future Improvements

- **Real-World Network Testing:** Scale to larger, more complex road networks using actual city traffic data
- **Efficiency Quantification:** Measure performance gains on realistic traffic patterns and network topologies
- **Visualization Component:** Add interactive traffic flow visualization to better understand agent behavior and network dynamics
- **Multi-Objective Optimization:** Balance travel time with other metrics like travel distance

## Research Context

Developed as part of the AI/ML REU program at Princeton University (Summer 2025) within the Reinforcement Learning lab. This work explores practical applications of multi-agent reinforcement learning to transportation optimization problems.

## Usage

```bash
# Train the agent
python train_agent.py

# Run baseline comparison
python traffic_baseline.py
```

## Dependencies

- `stable-baselines3`: PPO implementation
- `networkx`: Graph-based traffic network modeling
- `gymnasium`: RL environment framework
- `numpy`, `matplotlib`: Data processing and visualization
