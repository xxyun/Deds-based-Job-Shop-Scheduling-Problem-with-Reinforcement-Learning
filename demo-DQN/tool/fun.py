import json
import requests
from copy import deepcopy
from itertools import filterfalse
import numpy as np


def read_json(file_path):
    return json.load(open(file_path, 'r', encoding='utf-8'))


def write_json(write_data, file_path):
    return json.dump(write_data, open(file_path, 'w', encoding='utf-8'), ensure_ascii=False)


def compare(x, y, idx=0):
    # False if np.where(x <= y)[0].size > 0 else True
    if len(x) != len(y):
        return True
    if type(x) == dict:
        x = list(x.values())
    if type(y) == dict:
        y = list(y.values())
    if len(x) == 0:
        return False
    if idx == len(x):
        return False
    if y[idx] < x[idx]:
        return True
    elif y[idx] == x[idx]:
        return compare(x, y, idx=idx + 1)
    else:
        return False


def merge_time_intervals(intervals):
    if not intervals:
        return []
    intervals = filterfalse(lambda x: x[1] < 0, intervals)
    intervals = filterfalse(lambda x: x[0] >= x[1], intervals)
    sorted_intervals = sorted(intervals)
    merged_intervals = []
    for interval in sorted_intervals:
        if not merged_intervals or interval[0] > merged_intervals[-1][1]:
            merged_intervals.append([max(interval[0], 0), interval[1]])
        else:
            merged_intervals[-1] = [max(merged_intervals[-1][0], 0), max(merged_intervals[-1][1], interval[1])]
    return merged_intervals


def post_test(post_url, post_data):
    r = requests.post(post_url, data=json.dumps(post_data))
    output = json.loads(r.text)
    return output
