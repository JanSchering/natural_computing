from os import getcwd
from os.path import join
from typing import Tuple
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

tensorboard_dir = join(getcwd(), "TB")
flush_secs = 10

# ---------- Adjusted from https://github.com/rasmusbergpalm/vnca/blob/main/util.py

def get_writers() -> Tuple[SummaryWriter, SummaryWriter]:
    """
    Helper-Function: Creates two Tensorboard writers - one for the training logs and one for the test/validation logs.
    """
    train_writer = SummaryWriter(tensorboard_dir + 'train', flush_secs=flush_secs)
    test_writer = SummaryWriter(tensorboard_dir + 'test', flush_secs=flush_secs)
    return train_writer, test_writer
