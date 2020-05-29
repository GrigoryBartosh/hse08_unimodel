import os
import json
import datetime

from torch.utils.tensorboard import SummaryWriter

from common.config import PATH
import common.utils as utils

__all__ = ['get_summary_writer']


def write_arguments(summary_writer, args):
    s = json.dumps(args, indent=4)
    s = s.replace(' ', '&nbsp;')
    s = s.replace('\n', '  \n')
    summary_writer.add_text('args', s, )


def get_summary_writer(name=None):
    date = str(datetime.datetime.now())[:19]
    name = date + ' ' + name if name else date

    utils.make_dir(PATH['TF_LOGS'])
    logs_path = os.path.join(PATH['TF_LOGS'], name)
    return SummaryWriter(logs_path)
