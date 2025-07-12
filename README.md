# Enhanced PPO Mobile Robot Navigation

**Enhanced PPO Mobile Robot Navigation** is a research-oriented code base that demonstrates how far you can push *Proximal Policy Optimization* (PPO) for indoor, map-less mobile-robot navigation once you combine it with modern reinforcement-learning tricks such as Prioritized Experience Replay (PER), Conditional Value-at-Risk (CVaR), a dynamically tuned entropy bonus, and an adaptive curriculum.  
The target platform is the **Scout Mini** robot simulated in **Isaac Sim** (20 m × 20 m virtual warehouse with static obstacles), but the algorithm itself is environment-agnostic and can be ported to other ROS 2-based robots with minimal changes. :contentReference[oaicite:0]{index=0}

---

## Why this repository?

Conventional PPO is popular for continuous-control tasks, yet it still struggles with  
* **Sparse rewards** in large, cluttered spaces,  
* **Risk-blind optimisation** (it cares about the *mean* return, not the *tail* risk), and  
* **Slow sample efficiency** in physics-accurate simulators.

This repo shows a carefully engineered recipe that overcomes those limitations **without drifting away from PPO’s simple, on-policy core**.

---

## Key ingredients

| Module | Purpose | How it is realised in the code |
|--------|---------|--------------------------------|
| **Adaptive Curriculum** | Start easy, end hard | `GoalNavEnv.curr_expand_step` gradually widens the goal-sampling annulus |
| **PER for PPO** | Re-use the most informative transitions even in an on-policy loop | `replay_buffer.py` implements a SumTree buffer plus IS weights |
| **Dynamic Entropy Schedule** | Keep exploration high when it matters, decay it when the agent is confident | `agent.py` updates β with distance- & clutter-aware terms |
| **CVaR Regulariser** | Penalise the worst-α % returns → safer policies | `train.py` augments the PPO loss with a CVaR penalty |
| **Two-headed Actor-Critic** | Stable updates, quick inference | `model.py` defines separate actor / critic MLPs |

---

## Code layout
enhanced-ppo-mobile-robot-nav/
├─ goal_nav_env.py # Isaac Sim + ROS 2 environment wrapper
├─ model.py # Actor / Critic network definitions
├─ agent.py # PPO algorithm with PER, CVaR, entropy schedule
├─ replay_buffer.py # Prioritised replay buffer (SumTree)
├─ train.py # From-scratch training script
├─ train_continue.py # Resume-training utility
└─ utils.py # Logging, GAE, etc.

---

## Files

- **`goal_nav_env.py`** — Isaac Sim + ROS 2 environment wrapper  
- **`model.py`** — Actor-critic network definitions  
- **`agent.py`** — PPO agent with Prioritized Experience Replay, CVaR regularisation, and a dynamic entropy schedule  
- **`replay_buffer.py`** — Sum-tree replay buffer for on-policy PER  
- **`train.py`** — Main training script (from scratch)  
- **`train_continue.py`** — Resume training from a saved checkpoint  
- **`utils.py`** — Helper functions (GAE, logging, curriculum control, …)  
- **`__init__.py`** — Package initialiser  

---

## Note

URDF/USD map files, the Scout Mini robot model, and the Isaac Sim world are assumed to be pre-configured in the user’s environment.
