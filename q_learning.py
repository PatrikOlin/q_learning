import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
state = env.reset()
DISCRETE_OS_SIZE = [20,20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# Q-learning settings
LEARNING_RATE = 0.3
DISCOUNT = 0.95
EPISODES = 10000

SHOW_EVERY = 1000
STATS_EVERY = 100

# For stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

# Exploration settings
epsilon = 0.50
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            # get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()
            # print(episode_reward)

        # If simulation did not end yet after last sted - update Q table
        if not done:

            # Maximum possible Q values in next step
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state +  (action,)]

            # ANd heres our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q table
            q_table[discrete_state + (action,)] = new_q

            # Simulationen ended (for any reason) - if goal position is achieved - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            #q_table[discrete_state + (action,)] = reward
            print(f'Made it on episode {episode}')
            q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

    #Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/len(ep_rewards[-STATS_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average: {average_reward:>4.1f}, min: {min(ep_rewards[-STATS_EVERY:])}, max: {max(ep_rewards[-STATS_EVERY:])}, current epsilon: {epsilon:>1.2f}')

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='average rewards')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max rewards')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min rewards')
plt.legend(loc=2)
plt.grid(True)
plt.show()
