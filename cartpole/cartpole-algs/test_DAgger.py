import gymnasium as gym
import torch
import torch.nn as nn
from ppo import ENV_ID, DEVICE

obs_dim = 4
act_dim = 2

class StudentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

env = gym.make(ENV_ID, render_mode="human")
obs, _ = env.reset(seed=0)

student_net = StudentNet(obs_dim, act_dim).to(DEVICE)
student_net.load_state_dict(torch.load("dagger_student.pth"))
student_net.eval()

done = False
ep_return = 0
while not done:
    x = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = student_net(x)
        action = torch.argmax(logits).item()
    obs, reward, terminated, truncated, _ = env.step(action)
    ep_return += reward
    done = terminated or truncated

print("Student episode return:", ep_return)
env.close()
