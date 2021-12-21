import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')
#env._max_episode_steps = 500
env.reset()
episode_finish_times = np.array([])

#print(env.reset())

#if pole angle < 0 push cart left, if pole angle > 0 push right
def naive_controller(observation):
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

#Other state-space eqn based control algs: Kalman filter, robust, MPC

for episode in range(5):
    observation = env.reset()
    #print(type(observation))
    #print(observation)

    #target = np.array([0.6, 0]) for MountainCar-v0
    target = np.array([0, 0, 0, 0])
    errors = [observation - target]
    actions = []
    for t in range(500):
        env.render()
       
        #Baseline Naive Strategy
        #action = naive_controller(observation)
        
        #PID
        #print("Test: ", observation)
        errors.append(observation - target)
        action = PID(errors)
        actions.append(action)

        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            episode_finish_times = np.append(episode_finish_times,t+1)
            plt.plot(range(t+1), actions)
            plt.title("Force applied over timesteps t")
            plt.xlabel('t')
            plt.ylabel('Applied Force')
            plt.show()
            break

print("Average finish time: ", np.mean(episode_finish_times))
print(episode_finish_times)
env.close()

    

