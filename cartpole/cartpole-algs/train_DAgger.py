import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ppo import ActorCritic, DEVICE, ENV_ID

#initial the simulation environment
env = gym.make(ENV_ID)

#extract the dimensions of the observation and action spaces
obs_dim = env.observation_space.shape[0] #dim=1x4 4 features: cart position, cart velocity, pole angle, pole angular velocity
act_dim = env.action_space.n #dim=1x2 2 actions: move left or move right

#instantiate the actor-critic network
expert_net= ActorCritic(obs_dim, act_dim).to(torch.device(DEVICE))

# load pretrained weights for the expert policy
expert_net.load_state_dict(torch.load('ppo_cartpole.pth'))

#lock the expert network's parameters to prevent updates during training
expert_net.eval()  # set to evaluation mode

# Define the student network
class StudentNet(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(StudentNet,self).__init__()

        #a simple ffn
        self.network=nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,output_dim)
        )

    # define how data flows through the network
    def forward(self,x):
        return self.network(x)

#instantiate the student network
student_net=StudentNet(obs_dim,act_dim).to(DEVICE)

#setup the optimizer for the student network
optimizer=optim.Adam(student_net.parameters(),lr=1e-3)

#set up the loss function for training the student network
loss_fn=nn.CrossEntropyLoss()

# ==============================================================================
# DAGGER ALGORITHM FLOW & DATASET TENSOR EVOLUTION
# ==============================================================================
# Example Scenario Context: 
# - Environment: CartPole (Observation: 4D vector, Action: 1D scalar integer)
# - Settings: EPISODES_PER_ITER = 5
#
# FLOW & DIMENSIONAL VISUALIZATION:
# 
# Iteration 0 (Initialization):
# - Environment Interaction: Expert agent drives 5 perfect episodes for example each episode lastes 100 steps, so 500 steps total
# - Action Generation:       Expert agent provides actions
# - Data Collected:          500 steps of perfect trajectories
# - Master Dataset Status:   500 total pairs
# - Training Tensor Shapes:  obs_tensor    -> Shape: (500, 4)
#                            action_tensor -> Shape: (500,)
# - Training Phase:          Imitator trains on this (500, 4) tensor.
#
# Iteration 1 (The First Test & First Mistakes):
# - Environment Interaction: Imitator agent drives (clumsy, ~50 steps/ep = 250 steps total)
# - Action Generation:       Expert agent calculates correct actions for those 250 clumsy states
# - Data Collected:          250 steps of error-recovery trajectories
# - Master Dataset Status:   500 (Historical) + 250 (New) = 750 total pairs
# - Training Tensor Shapes:  obs_tensor    -> Shape: (750, 4)
#                            action_tensor -> Shape: (750,)
# - Training Phase:          Imitator retrains on the ENTIRE (750, 4) tensor.
#
# Iteration 2 (Expanding the Boundary):
# - Environment Interaction: Imitator drives again (improved, ~80 steps/ep = 400 steps total)
# - Action Generation:       Expert agent calculates correct actions for these new edge-cases
# - Data Collected:          400 steps of new edge-case trajectories
# - Master Dataset Status:   750 (Historical) + 400 (New) = 1150 total pairs
# - Training Tensor Shapes:  obs_tensor    -> Shape: (1150, 4)
#                            action_tensor -> Shape: (1150,)
# - Training Phase:          Imitator retrains on the ENTIRE (1150, 4) tensor.
#
# * Note: As iterations progress, the batch dimension (N, 4) continuously grows, 
#         forcing the imitator to map an increasingly diverse set of states to 
#         expert-approved actions.
# ==============================================================================

DAGGER_ITERATIONS = 8      # Total number of collection/training cycles
EPISODES_PER_ITER = 5      # Number of environment rollouts per cycle
EPOCHS_PER_UPDATE = 10     # Number of training passes over the master dataset per cycle
#3. DAgger training loop- Data Collection
DAGGER_ITERATIONS=8 # how many cycles of data collection and student training we perform in DAgger
EPISODES_PER_ITER=5 #how many episodes the student policy have to play in the env before it pauses to retrain on the aggregated dataset
EPOCHS_PER_UPDATE=10 # after student 5 episodes of interaction with the env, we retrain the student network for 10 epochs on the aggregated dataset of observations and expert actions

dataset_obs=[]
dataset_actions=[]

for iteration in range(DAGGER_ITERATIONS):
    for _ in range(EPISODES_PER_ITER):
        #reset the env for a new episode
        obs, _ = env.reset()
        done=False

        while not done:
            #convert the observation array into a batched pytorch tensor
            
            x=torch.from_numpy(obs).float.unsqueeze(0).to(DEVICE) #shape: [4,], unsqueeze add a dimension so shape becomes [1,4]

            #query the expert for the action to take in the current state
            with torch.no_grad():
                expert_logits, _ = expert_net(x)
                expert_actions=torch.argmax(expert_logits).item() #get the action with the highest logit score

            if iteration==0:
                #in the first iteration, we only collect data from the expert
                student_action=expert_actions

            else:
                #in subsequent iterations, we use the student policy to select actions
                with torch.no_grad():
                    student_logits=student_net(x)
                    student_actions=torch.argmax(student_logits).item()

            #aggregate the data: Student's observation and Expert's action
            dataset_obs.append(obs)
            dataset_actions.append(expert_actions)

            #step the environment using the student's action
            obs,reward,terminated,truncated,_=env.step(student_actions)
            done=terminated or truncated

    #4. retrain the student network on the aggregated dataset
    obs_tensor=torch.tensor(np.array(dataset_obs),dtype=torch.float32).to(DEVICE) #shape: [N,4]
    actions_tensor=torch.tensor(dataset_actions,dtype=torch.long).to(DEVICE) #shape: [N,1]

    student_net.tain() #set the student network to training mode

    for epoch in range(EPOCHS_PER_UPDATE):
        #clear the gradients
        optimizer.zero_grad()

        #forward pass student predicts actions for the aggregated observations
        student_logits=student_net(obs_tensor) #shape: [N,2]

        #compute the loss between the student's predictions and the expert's actions
        loss=loss_fn(student_logits,actions_tensor)

        #backpropagation and optimization step
        loss.backward()
        optimizer.step()



            
    
