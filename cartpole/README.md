# RL & Imitation Learning — Gymnasium

Train classic Gymnasium control tasks with clean, extensible RL and imitation learning code.

**Reinforcement learning:**
- ✅ PPO + CartPole-v1
- ✅ Tabular Q-learning + CartPole-v1
- ✅ Tabular SARSA + CartPole-v1
- ✅ DQN + CartPole-v1
- 🔜 A2C, SAC (drop-in structure ready)

**Imitation learning:**
- ✅ DAgger + CartPole-v1

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# train PPO on CartPole (MPS auto on Apple Silicon)
python scripts/train.py --env-id CartPole-v1 --algo ppo --total-steps 1000000 --save ppo_cartpole.pth

# watch the trained policy(same approach with other algs)
python scripts/eval.py --env-id CartPole-v1 --algo ppo --ckpt ppo_cartpole.pth
