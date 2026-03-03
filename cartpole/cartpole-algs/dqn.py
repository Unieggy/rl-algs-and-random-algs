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

# The Q network
class QNetwork(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super().__init__()

        #taks in state(4,),outputs expected score for each action (2,)
        self.net=nn.Sequential(
            nn.Linear(obs_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,act_dim) #no softmax
            )
        
        def forward(self,x):
            return self.net(x)
        

#the replay buffer
class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity) #automatically pushes old data out when full
    
    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    
    def sample(self,batch_size):
        #grab a random batch of memories
        batch=random.sample(self.buffer,batch_size)
        states,actions,rewards,next_states,dones=zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards,dtype=np.float32),
            np.array(next_states),
            np.array(dones,dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)
    

#training loop
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_speed(SEED)

    env=gym.make(ENV_ID)
    obs_dim=env.observation_space.shape[0]
    act_dim=env.action_space.n

    #create both networks
    policy_net=QNetwork(obs_dim,act_dim).to(DEVICE)
    target_net=QNetwork(obs_dim,act_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())# Make them identical to start
    target_net.eval()#target net never calculates gradients

    optimizer=optim.Adam(policy_net.parameters(),lr=LR)
    memory=ReplayBuffer(BUFFER_SIZE)

    state, _=env.reset(seed=SEED)
    ep_reward=0
    returns_log=[]

    for step in range(MAX_STEPS):
        #choose action (epsilon-greedy)
        #calculate current epsilon
        epsilon=EPSILON_END+(EPSILON_START-EPSILON_END)*np.exp(-1. * step/EPSILON_DECAY)

        if random.random()<epsilon:
            action=env.action_space.sample() #explore:random action
        else:
            with torch.no_grad():
                state_tensor=torch.FloatTensor(state).unsqueeze(0).to(DEVICE)# shape:(1,4)
                q_values=policy_net(state_tensor) # shape:(1,2)
                action=q_values.argmax().item() #Exploit: pick highest Q-value

        #2.step the environment
        next_state,reward,terminated,truncated,_=env.step(action) 
        done=terminated or truncated

        #3 store in replay buffer
        memory.push(state,action,reward,next_state,done)

        state=next_state
        ep_reward+=reward

        if done:
            returns_log.append(ep_reward)
            ep_reward=0
            state,_=env.reset()

        #train the network(if we have enough data)
        if len(memory)>=BATCH_SIZE:
            #sample a batch of 128 transitions
            b_states,b_actions,b_rewards,b_next_states,b_dones=memory.sample(BATCH_SIZE)

            # Convert to tensors
            b_states = torch.FloatTensor(b_states).to(DEVICE)          # (128, 4)
            b_actions = torch.LongTensor(b_actions).unsqueeze(1).to(DEVICE) # (128, 1)
            b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(DEVICE) # (128, 1)
            b_next_states = torch.FloatTensor(b_next_states).to(DEVICE)      # (128, 4)
            b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(DEVICE)     # (128, 1)


            # Compute Current Q-Values: Q(s, a)
            # We pass all 128 states to the network, but use .gather() to pluck out 
            # only the Q-value of the specific action we actually took in the past.
            # Row 1: State 1. Row 2: State 2. Row 3: State 3.
            # Columns are [Score for Left, Score for Right]
            # [[ 10.5,  12.0 ], [  8.0,   7.5 ],[ 15.0,  14.2 ]]
            current_q_values = policy_net(b_states).gather(1, b_actions)

            #compute target Q-values using the target network: r+gamma*max Q(s')
            with torch.no_grad():
                
                #[[  9.0,  14.0 ],
                # [  6.0,   5.0 ],
                # [ 18.0,  20.0 ]]
                #look across dimension 1 to find the max valude in each row
                next_q_values=target_net(b_next_states).max(1)[0].unsqueeze(1)
                target_q_values=b_rewards+(GAMMA*next_q_values*(1-b_dones))

            # compute loss (mean sqaured error)
            loss=nn.MSELoss()(current_q_values,target_q_values)

            #optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 5. Update Target Network
        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()
    print("Training complete. Last 10 episode returns:", returns_log[-10:])

if __name__ == "__main__":
    train()


