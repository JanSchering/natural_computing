import os 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

revision = "%s" % datetime.now()
tensorboard_dir = 'tensorboard'
flush_secs = 10

def get_writers(name):
    train_writer = SummaryWriter(os.path.join(os.getcwd(), name, 'train'), flush_secs=flush_secs)
    test_writer = SummaryWriter(os.path.join(os.getcwd(), name, 'test'), flush_secs=flush_secs)
    return train_writer, test_writer