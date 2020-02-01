import tensorflow as tf
import os


def find_latest_tfevents_file(dir_path):
    files = os.listdir(dir_path)
    files = list(filter(lambda x: x.find("tfevents") != -1, files))
    if len(files) == 0:
        return None
    return sorted(files)[-1]


def extract_tag_from_tensorboard(tb_events_file, tag):
    ret_events = []
    for event in tf.train.summary_iterator(tb_events_file):
        step = event.step
        for value in event.summary.value:
            if value.tag == tag:
                ret_events.append((step, value.simple_value))
    return ret_events
