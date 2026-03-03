#cartpole using dqn
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- Hyperparameters ---
ENV_ID="CartPole-v1"
BUFFER_SIZE     = 10_000     # Size of the Experience Replay
BATCH_SIZE      = 128        # How many memories to sample for one training step
GAMMA           = 0.99       # Discount factor
LR              = 1e-3       # Learning rate
TARGET_UPDATE   = 100        # How often (in steps) to copy weights to the Target Net
EPSILON_START   = 1.0        # 100% random actions at start
EPSILON_END     = 0.05       # 5% random actions at end
EPSILON_DECAY   = 1000       # How fast epsilon decays
MAX_STEPS       = 20_000     # Total training steps
SEED            = 42
DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"