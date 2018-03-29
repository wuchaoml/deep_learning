import gym
import time
from RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000, e_greedy_increment=0.0001,)

total_steps = 0
time_total = []

for i_episode in range(30):
    observation = env.reset()
    ep_r = 0
    start_time = time.time()
    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        position, velocity = observation_
        reward = abs(position - (-0.5))
        RL.store_transition(observation, action, reward, observation_)
        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            get = '| Get' if observation_[
                0] >= env.unwrapped.goal_position else '| ----'
            print('| Ep_r:', round(ep_r, 4), '| Epsilon:', round(RL.epsilon, 2))

            duration_time = time.time() - start_time
            time_total.append(duration_time)
            time.sleep(1)
            break
        observation = observation_
        total_steps += 1
for i in range(len(time_total)):
    print(time_total[i])
RL.plot_cost()
