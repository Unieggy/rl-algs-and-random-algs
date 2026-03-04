import gymnasium as gym
import torch
import torch.nn as nn
import time

# 1. We must define the exact same network architecture to load the weights into
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)

def watch_agent_play(model_path="dqn_cartpole.pth", episodes=5):
    # 2. Initialize environment with render_mode="human" for visualization
    env = gym.make("CartPole-v1", render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 3. Instantiate the network and load the saved weights
    policy_net = QNetwork(obs_dim, act_dim).to(device)
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Could not find {model_path}. Make sure you saved it in your training script!")
        return

    # Put the network in evaluation mode (disables dropout/batchnorm if you had them)
    policy_net.eval()

    # 4. Play the game!
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        
        print(f"Starting Episode {ep + 1}...")
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # NO EPSILON-GREEDY HERE. We want pure exploitation.
            # We trust the network to always pick the action with the highest Q-value.
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            
            # Step the environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            
            # Add a tiny delay so the window doesn't flash by too fast to see
            time.sleep(0.02)
            
        print(f"Episode {ep + 1} finished with Total Reward: {ep_reward}")

    env.close()

if __name__ == "__main__":
    watch_agent_play()