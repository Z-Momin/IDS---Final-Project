import random
import numpy as np
import copy
from torch.utils.data import DataLoader

ACTION_SPACE = np.array([
    [0.0, 0.0, 0.0],        # Nothing
    [0.0, 1.0, 0.0],        # Accelerate
    [0.0, 0.0, 0.2],        # Brake
    [-1.0, 0.0, 0.0],       # Left
    [-1.0, 1.0, 0.0],       # Left, Accelerate
    [-1.0, 0.0, 0.2],       # Left, Brake
    [1.0, 0.0, 0.0],        # Right
    [1.0, 1.0, 0.0],        # Right, Accelerate
    [1.0, 0.0, 0.2]         # Right, Brake
])


def get_dl(train_ds, valid_ds, batch_size, **kwargs):
    return (DataLoader(train_ds, batch_size, shuffle=True, **kwargs),
            DataLoader(valid_ds, 2 * batch_size, **kwargs))


def lr_find(dl, model, loss_func, optimizer, max_iter=100, min_lr=1e-6, max_lr=10):
    print("... finding learning rate")
    n_iter = 0
    lrs = []
    losses = []
    init_st = copy.deepcopy(model.state_dict())
    while n_iter < max_iter:
        for i, (xb, yb) in enumerate(dl):
            if n_iter >= max_iter: break
            model.train()
            lr = min_lr * (max_lr / min_lr) ** (n_iter / max_iter)
            lrs.append(lr)
            for pg in optimizer.param_groups: pg['lr'] = lr
            y_pred = model(xb)
            loss = loss_func(y_pred, yb)
            losses.append(loss.item())            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter += 1 
            if n_iter % 10 == 0: print(f"...    {n_iter*100/max_iter:.0f}% complete")
    model.load_state_dict(init_st)
    return lrs, losses
    

def exponential_moving_average(x, beta=0.9):
        average = 0
        ema_x = x.copy()
        for i, o in enumerate(ema_x):
            average = average * beta + (1 - beta) * o
            ema_x[i] = average / (1 - beta**(i+1))
        return ema_x


def id_to_action(id):
    return ACTION_SPACE[id]


def rgb2gray(rgb):
    """ 
    This method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')


class FrameHistory(object):

    def __init__(self, history_length):
        self.history_length = history_length
        self.history = []

    def push(self, frame):
        """Saves a frame."""
        self.history.append(frame)
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def clone(self):
        new = FrameHistory(self.history_length)
        new.history = self.history.copy()
        return new

    def __len__(self):
        return len(self.history)