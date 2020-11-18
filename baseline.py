import numpy as np
import gym
import os
import json

from datetime import datetime
from data_collection import save_results
from itertools import count


RENDER = False
N_EPISODES = 15


def baseline_model(state):
    return np.array([0., 1., 0.])


def run_episode(env, model, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    state = env.reset()
    
    for step in count():
        
        action = model(state)

        next_state, reward, done, info = env.step(action)   
        episode_reward += reward       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped
    results_fn = "results_baseline-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    model = baseline_model

    episode_rewards = []
    for episode in range(N_EPISODES):
        episode_reward = run_episode(env, model, rendering=RENDER)
        episode_rewards.append(episode_reward)
        print("\n[EPISODE]: {}".format(episode))
        print("[AVERAGE EPISODE REWARD]: " + "{:0.2f}".format(np.array(episode_rewards).mean() if len(episode_rewards) != 0 else episode_reward))

    print("... saving results")
    save_results(episode_rewards, filename=results_fn)

    env.close()
    print('... finished')
