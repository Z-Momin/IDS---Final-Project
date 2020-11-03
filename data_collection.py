import gym
import random
import numpy as np 

from pyglet.window import key


RENDER = True

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

if __name__ == "__main__":

    env_name = "CarRacing-v0"
    env = gym.make(env_name).unwrapped
    restart = False

    state = env.reset()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    action = np.array([0., 0., 0.])

    while True:
        next_state, reward, done, info = env.step(action)
        if RENDER: env.render()
        state = next_state
        if done or restart: break

    env.close()
