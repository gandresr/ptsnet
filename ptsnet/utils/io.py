import os
import subprocess
import shutil

from pkg_resources import resource_filename

def run_shell(command):
    subprocess.run(command.split(' '))

def get_root_path():
    rpath = resource_filename(__name__, '')
    token = 'ptsnet'
    idx = rpath.rfind(token)
    return rpath[:idx+len(token)]

def get_examples_path():
    return os.path.join(get_root_path(), 'examples')

def get_example_path(example_name):
    ename = example_name
    if not example_name.lower().endswith('.inp'):
        ename = ename.upper()
        ename += '.inp'
    return os.path.join(get_examples_path(), ename)

def walk(folder_structure, root_path):
    paths = []

    if type(folder_structure) == dict:
        for ff in folder_structure:
            paths.extend(walk(folder_structure[ff], os.path.join(root_path, ff)))
    elif type(folder_structure) in (list, tuple):
        for ff in folder_structure:
            paths.extend(walk(ff, root_path))
    elif type(folder_structure) == int:
        new_root_path = os.path.join(root_path, str(folder_structure))
        paths.append(new_root_path)

    return paths

def get_workspaces():
    return os.listdir(os.path.join(get_root_path(), 'workspaces'))

def get_temp_folder():
    return os.path.join(get_root_path(), 'tmp')

def create_temp_folder():
    tmpdir = get_temp_folder()
    if os.path.exists(tmpdir): shutil.rmtree(tmpdir)
    os.makedirs(tmpdir)