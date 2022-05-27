import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

revision = os.environ.get("REVISION") or "%s" % datetime.now()
message = os.environ.get('MESSAGE')
tensorboard_dir = os.path.join(os.getcwd(), "TB")
flush_secs = 10


def get_writers(name):
    train_writer = SummaryWriter(tensorboard_dir + 'train', flush_secs=flush_secs)
    test_writer = SummaryWriter(tensorboard_dir + 'test', flush_secs=flush_secs)
    return train_writer, test_writer
