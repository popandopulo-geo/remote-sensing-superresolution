import os
from tensorboardX import SummaryWriter


class TrainClock(object):
    """ Clock object to track epoch and step during training
    """
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0

    def tick(self):
        self.minibatch += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def save(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
        }

    def load(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']


def exp_env(path):
    log_dir = os.path.join(path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    train_tb = SummaryWriter(os.path.join(log_dir, 'train'))
    test_tb = SummaryWriter(os.path.join(log_dir, 'test'))
    valid_tb = SummaryWriter(os.path.join(log_dir, 'valid'))
                            
    return train_tb, test_tb, valid_tb
    