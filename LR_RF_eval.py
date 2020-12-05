import numpy as np
import gym
import os
import json

from datetime import datetime
from data_collection import save_results
from itertools import count
from sklearn.ensemble import RandomForestClassifier

RENDER = False
N_EPISODES = 15


from joblib import dump, load

clf = load('ova1.joblib')
scaler = load(open('scaler.pkl', 'rb'))

ACTION_SPACE = np.array([
    [0.0, 0.0, 0.0],        # Nothing
    [0.0, 1.0, 0.0],        # Accelerate
    [0.0, 0.0, 0.2],        # Brake
    [1.0, 0.0, 0.0],        # Right
    [-1.0, 0.0, 0.0],       # Left
    [1.0, 1.0, 0.0],        # Right, Accelerate
    [1.0, 0.0, 0.2],        # Right, Brake
    [-1.0, 1.0, 0.0],       # Left, Accelerate
    [-1.0, 0.0, 0.2]        # Left, Brake
])

def id_to_action(id):
    return ACTION_SPACE[id]

def frame_to_arr(frame):
  val = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
  return np.reshape(val, (96**2))

def baseline_model(state):
    gstate = frame_to_arr(state)
    gstate = gstate.reshape(1,-1)
    scaled_frame = scaler.transform(gstate)
    id = clf.predict(scaled_frame)

    return np.array(id_to_action(id[0]))


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
