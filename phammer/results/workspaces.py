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
        shutil.rmtree(os.path.join(get_root_path(), 'workspaces', d))

def list_workspaces():
    return os.listdir(os.path.join(get_root_path(), 'workspaces'))

def print_workspaces():
    txt = {}
    for d in list_workspaces():
        with open(os.path.join(get_root_path(), 'workspaces', d, 'settings.pkl'), 'rb') as f:
            s = pickle.load(f)
            txt[d] = f"T = {s['duration']}, t = {s['time_step']}, N = {s['num_points']}, p = {s['num_processors']}"
    print('\n')
    for i, d in enumerate(txt):
        print(f'  ({i})', d)
        print('    ' + txt[d])
        print('\n')

def num_workspaces():
    return len(list_workspaces())

def generate_workspace_name():
    w_name = str(hash(str(datetime.now())))
    w_name = w_name.replace('-', 'm')
    return 'W640465'