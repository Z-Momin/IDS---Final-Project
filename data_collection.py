import os
import gym
import random
import json 
import gzip
import pickle
import numpy as np 

from itertools import count
from pyglet.window import key
from datetime import datetime


RECORD = True


def key_press(k, mod):
    global restart, action
    if k == key.ENTER: restart   = True
    if k == key.LEFT: action[0]  = -1.0
    if k == key.RIGHT: action[0] = 1.0
    if k == key.UP: action[1]    = 1.0
    if k == key.DOWN: action[2]  = 0.2


def key_release(k, mod):
    global action
    if k == key.LEFT and action[0] == -1.0: action[0]  = 0.0
    if k == key.RIGHT and action[0] == 1.0: action[0] = 0.0
    if k == key.UP: action[1]    = 0.0
    if k == key.DOWN: action[2]  = 0.0


def store_data(data, filename='data.pkl.gzip', directory="./data"):
    # save data
    if not os.path.exists(directory):
        os.mkdir(directory)
    data_file = os.path.join(directory, filename)
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_results(episode_rewards, filename="results_manually.json", directory="./results"):
    if len(episode_rewards) == 0: return

    # save results
    if not os.path.exists(directory):
        os.mkdir(directory)

    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
    
    fname = os.path.join(directory, filename)
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')


def push_to_buffer(replay_buffer, state, action, next_state, reward, done):
    replay_buffer["state"].append(state)
    replay_buffer["action"].append(np.array(action))
    replay_buffer["next_state"].append(next_state)
    replay_buffer["reward"].append(reward)
    replay_buffer["terminal"].append(done)


if __name__ == "__main__":

    results_fn = "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    data_fn = "data-%s.pkl.gzip" % datetime.now().strftime("%Y%m%d-%H%M%S")

    env_name = "CarRacing-v0"
    env = gym.make(env_name).unwrapped
    restart = False

    replay_buffer = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }

    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    action = np.array([0., 0., 0.], dtype=np.float32)
    episode_rewards = []
    iteration = 0

    for episdoe in count():

        episode_reward = 0
        state = env.reset()

        while True:

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            push_to_buffer(replay_buffer, state, action, next_state, reward, done)

            state = next_state
            iteration += 1

            if iteration % 500 == 0 or done:
                print("\n[ITERATION]: {}".format(iteration))
                print("\n[EPISODE]: {}".format(episdoe))
                print("[AVERAGE EPISODE REWARD]: " + "{:0.2f}".format(np.array(episode_rewards).mean() if len(episode_rewards) != 0 else episode_reward))

            if RECORD and iteration % 100 == 0:
                print('... saving data')
                store_data(replay_buffer, data_fn, "./data")
                save_results(episode_rewards, results_fn,"./results")

            env.render()            
            if done or restart: break
            
        if restart: break

        episode_rewards.append(episode_reward)

    env.close()
