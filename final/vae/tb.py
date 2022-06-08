from os import getcwd
from os.path import join
from typing import Tuple
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

revision = "%s" % datetime.now()
tensorboard_dir = 'tensorboard'
flush_secs = 10

# ---------- Based off https://github.com/rasmusbergpalm/vnca/blob/main/util.py

def get_writers(name:str) -> Tuple[SummaryWriter, SummaryWriter]:
    """
    Helper-Function: Creates two Tensorboard writers - one for the training logs and one for the test/validation logs.
    """
    train_writer = SummaryWriter(join(getcwd(), name, 'train'), flush_secs=flush_secs)
    test_writer = SummaryWriter(join(getcwd(), name, 'test'), flush_secs=flush_secs)
    return train_writer, test_writer