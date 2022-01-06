# CartPole-v0 Agents
## The Environment
A pole is attached to a cart on a frictionless track. An agent is given
information on pole angle and angular velocity, and the carts' position
and velocity in the x-direction at every timestep. At every timestep an agent
can apply a leftwards or rightwards force on the cart. The objective is to
develop an approach for selecting the best action at every timestep.

## Agents
Naive, PD controller, and Deep Q Network (DQN) approaches have been developed
for guiding the actions of the agent at every timestep.

**Naive**: At every timestep, move the cartpole left if the pole is leaning
rightwards, and vice versa.

**PD Controller**: This computes an error term at every timestep, and uses this
in a closed loop feedback mechanism to choose the agents' action based on the
current value of the error (P), and the approximate rate of change of error (D).

**DQN**: Applies [Deep Q-Learning with Experience Replay](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
to guide agent decisions. 

## Model Performance
Naive:
PD: Achieves average score of 200
DQN: Achieves average score of 21 in 150 episodes
