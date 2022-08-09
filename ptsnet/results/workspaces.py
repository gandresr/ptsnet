import shutil
import os
import pickle
import pathlib
import uuid

from datetime import datetime
from ptsnet.utils.io import get_root_path
from ptsnet.utils.data import is_array

def get_num_tmp_workspaces():
    return len(list_workspaces())

def get_count_of_tmp_workspaces():
    return os.path.join(os.getcwd(), 'workspaces', 'count.pkl')

def new_workspace_name(is_root = True):
    if is_root:
        if not os.path.exists(get_count_of_tmp_workspaces()):
            count = 0
        else:
            with open(get_count_of_tmp_workspaces(), 'rb') as f:
                count = pickle.load(f)
        count += 1
        with open(get_count_of_tmp_workspaces(), 'wb') as f:
            pickle.dump(count, f)
        return f'W{count}'
    return None

def new_uuid_workspace_name(size=1):
    wnames = []
    for i in range(size):
        wname = uuid.uuid4().hex[:8]
        while os.path.isdir(os.path.join(os.getcwd(), 'workspaces', wname)):
            wname = uuid.uuid4().hex[:8]
        wnames.append(f"W{wname}")
    return wnames

def create_workspaces_folder(root=False):
    ws_path = os.path.join(os.getcwd(), 'workspaces')
    if root:
        if not os.path.exists(ws_path):
            os.mkdir(ws_path)
    return ws_path

def get_workspaces():
    return os.listdir(os.path.join(os.getcwd(), 'workspaces'))

def get_tmp_folder(root=False):
    return os.path.join(create_workspaces_folder(root), 'tmp')

def create_temp_folder(root=False):
    tmpdir = os.path.join(create_workspaces_folder(root), 'tmp')
    if root:
        if os.path.exists(tmpdir): shutil.rmtree(tmpdir)
        os.makedirs(tmpdir)
    return tmpdir

def list_workspaces():
    wps = [d for d in os.listdir(os.path.join(os.getcwd(), 'workspaces')) if '.' not in d].sort()
    return wps

def print_workspaces(full_path = False):
    txt = {}
    for d in list_workspaces():
        abs_path = os.path.join(os.getcwd(), 'workspaces', d)
        with open(os.path.join(abs_path, 'settings.pkl'), 'rb') as f:
            s = pickle.load(f)
        with open(os.path.join(abs_path, 'fname.pkl'), 'rb') as f:
            fname = pickle.load(f)
            if full_path:
                fname = os.path.basename(fname)
        txt[d] = [
            fname,
            f"T = {s['duration']}, t = {s['time_step']}, N = {s['num_points']}, n = {s['num_processors']}",
        ]
    print('\n')
    for i, d in enumerate(txt):
        print(f'  ({i})', txt[d][0])
        print('    ' + txt[d][1])
        print('\n')

def delete_all_workspaces():
    confirmation = None
    while confirmation is None:
        confirmation = input('Are you sure that you want to delete all the workspaces? (yes/no): ')
        if not confirmation in ('yes', 'no'):
            confirmation = None
    if confirmation == 'no':
        return
    ROOT = get_root_path()
    for d in os.listdir(os.path.join(os.getcwd(), 'workspaces')):
        abs_path = os.path.join(os.getcwd(), 'workspaces', d)
        if os.path.isdir(abs_path):
            shutil.rmtree(abs_path)
        else:
            os.remove(abs_path)

def delete_workspace(workspace_name, full_path=True, verbose=True):
    confirmation = None

    ROOT = get_root_path()
    wpl = list_workspaces()
    wnamelist = workspace_name

    txt = {}

    wnames = workspace_name
    if not is_array(workspace_name):
        wnames = [workspace_name]
        apaths = [os.path.join(os.getcwd(), 'workspaces', workspace_name)]
    else:
        apaths = \
            [os.path.join(
                os.getcwd(),
                'workspaces',
                wp_name) for wp_name in workspace_name]

    for i, apath in enumerate(apaths):
        f = pathlib.Path(apath)
        st = f.stat().st_mtime
        with open(os.path.join(apath, 'settings.pkl'), 'rb') as f:
            s = pickle.load(f)
        with open(os.path.join(apath, 'fname.pkl'), 'rb') as f:
            fname = pickle.load(f)
        if not full_path:
            fname = os.path.basename(fname)
        txt = [
            f'Last modification on: {datetime.fromtimestamp(st)}',
            fname,
            f"T = {s['duration']}, t = {s['time_step']}, N = {s['num_points']}, n = {s['num_processors']}",
        ]

        if verbose:
            print('\n')
            print(f'  ({wnames[i]}) ' + txt[0])
            print('      ' + txt[1])
            print('      ' + txt[2])
            print('\n')

            confirmation = None
            while confirmation is None:
                confirmation = input('Are you sure that you want to delete this workspace ? (yes/no): ')
                if not confirmation in ('yes', 'no'):
                    confirmation = None
        else:
            confirmation = 'yes'

        if confirmation == 'no': continue

        if os.path.isdir(apath):
            shutil.rmtree(apath)
        else:
            os.remove(apath)
