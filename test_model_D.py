import numpy as np
import gym
import os
import json
import torch

from datetime import datetime
from data_collection import save_results
from itertools import count

from torchvision import transforms as T
from model import Model_v2
from utils import rgb2gray, FrameHistory


RENDER = False
HISTORY_LENGTH = 5
N_EPISODES = 15

def id_to_action(id):
    steer, accelerate = id
    a = np.array([0., 0., 0.])
    if steer == 1: a[0] = -1.
    if steer == 2: a[0] = 1.
    if accelerate == 1: a[1] = 1.
    if accelerate == 2: a[2] = -0.2
    return a

def update_state(state, frame):
    state.push(transform_frame(frame))
    while len(state) < HISTORY_LENGTH:
        state.push(transform_frame(frame))


def select_action(model, state):
    """
    This mehtod selects an action based on the state per the policy network.
    """
    model.eval()
    with torch.no_grad():
        out = model(transform(torch.stack(state.history))[None, ...])
        steer = out[0].max(1).indices
        accelerate = out[1].max(1).indices
    return id_to_action([steer, accelerate])


def run_episode(env, model, state, rendering=RENDER, max_timesteps=1000):
    
    episode_reward = 0
    frame = env.reset()
    
    for step in count():
        
        update_state(state, frame)
        action = select_action(model, state)
        next_frame, reward, done, info = env.step(action)
        episode_reward += reward       
        frame = next_frame
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


def transform_frame(frame):
    return torch.from_numpy(rgb2gray(frame.astype("float32"))).to(device)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_fn = "model_D.pth"
    model_dir = "./models/"
    results_fn = "results_%s-%s.json" % (model_fn.split(".")[0], datetime.now().strftime("%Y%m%d-%H%M%S"))
    model = Model_v2([16, 32, 64, 64], nh=50, c_in=HISTORY_LENGTH).to(device)
    st = torch.load(model_dir + model_fn, map_location=device)
    model.load_state_dict(st)

    _m = torch.tensor([141.98462, 141.98868, 141.99257, 141.9944 , 141.9963 ])
    _s = torch.tensor([61.780296, 61.775593, 61.771965, 61.769146, 61.76684])

    transform = T.Compose([
                        # T.Lambda(lambda x: torch.from_numpy(rgb2gray(x.astype("float32"))).to(device)[None,...]),
                        T.Normalize(mean=_m, std=_s),
                        ])

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    state = FrameHistory(HISTORY_LENGTH)
    for episode in range(N_EPISODES):
        episode_reward = run_episode(env, model, state, rendering=RENDER)
        episode_rewards.append(episode_reward)
        print("\n[EPISODE]: {}".format(episode))
        print("[AVERAGE EPISODE REWARD]: " + "{:0.2f}".format(np.array(episode_rewards).mean() if len(episode_rewards) != 0 else episode_reward))
        env.close()

    print("... saving results")
    save_results(episode_rewards, filename=results_fn)
    print('... finished')
