import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')
#env._max_episode_steps = 500
env.reset()
episode_finish_times = np.array([])

def naive_controller(observation):
    '''
    if pole angle < 0 push cart left, if pole angle > 0 push right
    '''
    if observation[3] < 0:
        action = 0
    elif observation[3] > 0:
        action = 1
    return action

def sigmoid(z):
    return 1/(1 + np.exp(-z))
def PID(errors):
    K_p = 10 #5 # -0.15
    K_i = 0 #2 # -0.00001
    K_d = 200 #1 # -3

    action = K_p*errors[-1] + K_i*(np.sum(errors)) + K_d*(errors[-1] - errors[-2])

    #print("action: ", action)

    action = sigmoid(action[2])
    if action >= 0.5:
        action = 1
    else:
        action = 0
    return int(action)

# Other state-space eqn based control algs: Kalman filter, robust, MPC

class QNet(nn.Module):
    def __init__(self, num_states=4, num_actions=2):
        super(QNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_states,30),
            nn.ReLU(),
            nn.Linear(30,30),
            nn.ReLU(),
            nn.Linear(30,num_actions))

    def __call__(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
            x = torch.unsqueeze(x, dim=0)
        x = self.linear_relu_stack(x)
        return x
model = QNet()

def loss_fcn(y, Q):
    return nn.functional.mse_loss(y, Q)#.mean()

predict = lambda x: torch.argmax(nn.functional.softmax(x,dim=0))
opt = torch.optim.RMSprop(model.parameters(),lr=0.001)

def update_weights(X, Y):
    '''
    input: nparray of indices of replay memory D to use as training samples
    This updates Q Network parameters via gradient descent step
    '''
    opt.zero_grad()
    loss = loss_fcn(X, Y)
    loss.backward()
    opt.step()

#modularize this -> getting observations
M = 150
T = 500
D = [] # should be pooled over many episodes
model = QNet()
for episode in range(M):
    if episode > 0 and episode % 20 == 0:
        print("Episode {}:".format(episode), np.mean(episode_finish_times[-20:-1]))
    observation = env.reset()
    epsilon = 1
    #print(type(observation))
    #print(observation)
    #target = np.array([0, 0, 0, 0]) # target = np.array([0.6, 0]) for MountainCar-v0, this line not relevant to Q learning
    #errors = [observation - target] # this line not relevant to Q learning
    #actions = [] # this line not relevant to Q Learning
    for t in range(T):
        env.render()
        #print("Prediction: ", model(observation), model(observation).shape)
        
        # Baseline Naive Strategy
        #action = naive_controller(observation)
        
        # PID
        #print("Test: ", observation)
        #errors.append(observation - target)
        #action = PID(errors)
        #actions.append(action)

        #Q-Learning
        experience = [observation]
        epsilon = max(epsilon*0.995, 0.05)
        action_type = np.random.choice([0,1],p=[epsilon,1-epsilon])
        if action_type == 0:
            # do random action
            action = env.action_space.sample()
        else:
            # do action with greatest Q-value
            #print("DOING Q VALUE: ", predict(model(observation)))
            action = int(predict(model(observation))) # need to cast tensor to int

        observation, reward, done, info = env.step(action)
        experience.extend([action, reward, observation])
        #print("Experience: ", experience)
        D.append(experience)
        #print("Experience Replay: ", D, D[0][0])

        batch_size = 32
        gamma = 0.9
        train_indices = np.random.choice(np.arange(len(D)),size=batch_size)
        #print(train_indices, len(D))

        '''
        Debugging Q_values
        for i in train_indices:
            print(i, model(D[i][0]), D[i][1])#[D[i][1]]) #D[i][0], D[i][1], model(D[i][0]))
            print(model(D[i][0])[0][D[i][1]])

        Debugging Y_values
        for i in train_indices:
            print(i, D[i][2], gamma*max(model(D[i][3])[0]))

        Lesson: Need to access model()[0][index you want] NOt just model()[index you want]
        '''

        train_samples = [D[i] for i in train_indices]
        #print("D: {} \n D[0]: {}".format(D, D[0]))
        Q_values = [model(D[i][0])[0][D[i][1]] for i in train_indices] # list of tensors
        Y_values = [D[i][2] + gamma*max(model(D[i][3])[0]) for i in train_indices] # list of tensors
        #print("Q VALS: ", Q_values, "Y_VALS: ", Y_values)
        Q_values, Y_values = torch.stack(Q_values), torch.stack(Y_values)
        #print("Q VALS: ", Q_values, "Y_VALS: ", Y_values)
        update_weights(Q_values, Y_values)

        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            episode_finish_times = np.append(episode_finish_times,t+1)
            #print(t+1)
            #plt.plot(range(t+1), actions)
            #plt.title("Force applied over timesteps t")
            #plt.xlabel('t')
            #plt.ylabel('Applied Force')
            #plt.show()
            break

print("Average finish time: ", np.mean(episode_finish_times))
print(episode_finish_times)
plt.plot(episode_finish_times[::10])
plt.xlabel("Episode #")
plt.ylabel("Timesteps to termination")
plt.show()
env.close()
