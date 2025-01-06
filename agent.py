import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
from torch import nn
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime, timedelta

# PRINTING DATE - TIME
DATE_FORMAT = "%m-%d %H:%M:%S"

# DIRECTORY FOR SAVING RUN INFO
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class Agent:

    def __init__(self , hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_set[hyperparameter_set]
        
        self.hyperparameter_set = hyperparameter_set
        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.env_make_params    = hyperparameters.get('env_make_params' ,{})
        self.network_sync_rate  = hyperparameters['network_sync_rate']

        self.loss_fn = nn.MSELoss()    
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR , f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR , f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR , f'{self.hyperparameter_set}.png')

    def run(self , is_training=True , render = False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
            
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None , use_lidar=False)
        env = gymnasium.make(self.env_id, render_mode="human" if render else None , **self.env_make_params)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions , self.fc1_nodes).to(device)
        
        rewards_per_episode = []
        if is_training:

            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init
            
            # ADAM IS TYPE OF OPTIMIZER MOST PROBABLY USED
            self.optimizer = torch.optim.Adam(policy_dqn.parameters() , lr = self.learning_rate_a)

            # to make target and policy same initially copying all weight and biases
            target_dqn = DQN(num_states,num_actions,self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # track number of steps taken....Used for syncing policy -> target network
            step_count = 0

            # LIST TO KEEP TRACK OF EPSILON DECAY
            epsilon_history = []

            # TRACK BEST REWARD
            best_reward = -9999999

        else:
            # LOAD LEARNED POLICY
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # SWITCH MODEL TO EVALUATION MODE
            policy_dqn.eval()


        for episode in itertools.count():

            state,_ =env.reset()
            # THINGS WHICH ARE GOING IN NETWORK .. SHOULD BE CONVERTED TO TENSORS(tensor is n dimensional array)
            state = torch.tensor(state , dtype=torch.float32 , device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                # Next action:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action , dtype=torch.int64 , device=device)
                else:
                    with torch.no_grad(): # pytorch DOES GRADIENT ON IT'S OWN SO THIS LINE TELL TO DONT DO GRADIENT
                        # returns the index of the maxium q value among the all q values of output layer of policy network

                        # tensor([1,2,3,....]) --> tensor([[1,2,3,...]])
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                episode_reward += reward
                new_state = torch.tensor(new_state , dtype=torch.float32 , device=device)
                reward = torch.tensor(reward , dtype=torch.float32 , device=device)
                
                if is_training:
                    memory.append((state, action,new_state, reward, terminated))

                    # INCREMENT step_count
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)
            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                
                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)             
                    last_graph_update_time = current_time

                # CHECK IF MEMORY OF REPLAY_MEMORY ACHIEVED MINIMUM SIZE OF batch_size
                if len(memory) > self.mini_batch_size:

                    # SOME SAMPLE IS TAKEN FROM MEMORY
                    mini_batch = memory.sample(self.mini_batch_size)
                    # optimizing the target and policy network from the sample collected mini_batch
                    self.optimize(mini_batch,policy_dqn,target_dqn)

                    epsilon = max(epsilon*self.epsilon_decay , self.epsilon_min)  
                    epsilon_history.append(epsilon)  
                    
                    # after certain number of steps COPY POLICY NETWORK TO TARGET NETWORK
                    if(step_count > self.network_sync_rate):
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0


    def save_graph(self , reward_per_episode , epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs epsiodes (X-axis)
        mean_reward = np.zeros(len(reward_per_episode))
        for x in range(len(mean_reward)):
            mean_reward[x] = np.mean(reward_per_episode[max(0,x-99):(x+1)])
        plt.subplot(121)
        # plt.xlabel('Episodes')
        plt.ylabel('mean_reward')
        plt.plot(mean_reward)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)
        # plt.xlabel('Episodes')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0 , hspace=1.0)

        # SAVE PLOT
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    # Calculate the target (means calculate q value then set it in target network) and train the policy (then use target network to train policy network)
    def optimize(self , mini_batch , policy_dqn , target_dqn):
        
        # # THIS DOES COMPUTATION ONE-BY-ONE AMONG THE 32 TUPLES:-
        # # # we LOOK through the experiences in mini_batch and train target network on it(past experiences)
        # for state , action , new_state , reward , terminated in mini_batch:
        #     # USING Q FORMULAE OF DEEP Q NETWORK
        #     if terminated:
        #         # q-value for the output layer of target network
        #         target_q = reward
        #     else:
        #         with torch.no_grad():
        #             # q-value for the output layer of target network
        #             target_q =reward+self.discount_factor_g*target_dqn(new_state).max()

        #     current_q = policy_dqn(state)
# -----------------------------------------------------------------------------------------
            # # WE CAN COMPUTE THOSE 32 TUPLE AT ONCE
            states , actions , new_states , rewards , terminations = zip(*mini_batch)

            # stack creates batch tensors
            states = torch.stack(states)
            actions = torch.stack(actions)
            new_states = torch.stack(new_states)
            rewards = torch.stack(rewards)
            terminations=torch.tensor(terminations).float().to(device) 

            with torch.no_grad():
                # only 1 formulae to calculate Q as (1-terminations) handle the case when termination is true menas 1 so 1-1 =0 and target_q = reward
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]                
                '''
                    target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
                '''
            current_q = policy_dqn(states).gather(dim=1 , index = actions.unsqueeze(dim=1)).squeeze()
# ------------------------------------------------------------------------------------------------------
            #Compute the loss for the sample from mini_batch
            loss = self.loss_fn(current_q,target_q)

            # Optimize model
            # THESE 3 LINES MEANS:-
            # MAKING THE GRADIENT DESCENT BETWEEN THE LOSS AND WEIGHT ... AS OUR AIM IS TO MINIMIZE THE LOSS SO WE KEEP ON ADJUSTING WEIGHT...AS OUR PREDICTION(Y) OF NODE HAS A LINEAR EQUATION SO PREDICTION VALUE DEPENDS ON WEIGHT AND BIAS BOTH...HERE ALSO WE HAVE learning_rate WHICH DETERMINES HOW HUGE STEP WE TAKE TO CHANGE WEIGHT EACH TIME
            self.optimizer.zero_grad()      
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    # PARSE COMMAND LINE INPUT (PARSE IS ANOTHER METHOD OF TAKING INPUTS FROM USER/FROM CODE)
    parser = argparse.ArgumentParser(description="Train or Test model")
    parser.add_argument('hyperparameters' , help='')
    parser.add_argument('--train' , help='Training mode' , action='store_true') # --train means it is optional
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False,render=True)
