import shutil
import os
import pickle

from datetime import datetime
from phammer.utils.io import get_root_path

def delete_all_workspaces():
    confirmation = None
    while confirmation is None:
        confirmation = input('Are you sure that you want to delete all the workspaces? (yes/no): ')
        if not confirmation in ('yes', 'no'):
            confirmation = None
    if confirmation == 'no':
        return
    ROOT = get_root_path()
    for d in os.listdir(os.path.join(ROOT, 'workspaces')):
        abs_path = os.path.join(get_root_path(), 'workspaces', d)
        if os.path.isdir(abs_path):
            shutil.rmtree(abs_path)
        else:
            os.remove(abs_path)

def list_workspaces():
    wps = [d for d in os.listdir(os.path.join(get_root_path(), 'workspaces')) if '.' not in d]
    wps.sort(key = lambda x: int(x[1:]))
    return wps

def print_workspaces(full_path = False):
    txt = {}
    for d in list_workspaces():
        abs_path = os.path.join(get_root_path(), 'workspaces', d)
        with open(os.path.join(abs_path, 'settings.pkl'), 'rb') as f:
            s = pickle.load(f)
        with open(os.path.join(abs_path, 'fname.pkl'), 'rb') as f:
            fname = pickle.load(f)
            if full_path:
                fname = os.path.basename(fname)
        txt[d] = [fname, f"T = {s['duration']}, t = {s['time_step']}, N = {s['num_points']}, n = {s['num_processors']}"]
    print('\n')
    for i, d in enumerate(txt):
        print(f'  ({i})', txt[d][0])
        print('    ' + txt[d][1])
        print('\n')

def num_workspaces():
    return len(list_workspaces())

def get_count_path():
    return os.path.join(get_root_path(), 'workspaces', 'count.pkl')

def new_workspace_name(is_root = True):
    if is_root:
        if not os.path.exists(get_count_path()):
            count = 0
        else:
            with open(get_count_path(), 'rb') as f:
                count = pickle.load(f)
        count += 1
        with open(get_count_path(), 'wb') as f:
            pickle.dump(count, f)
        return f'W{count}'
    return None