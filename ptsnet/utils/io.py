import os, subprocess, shutil, datetime
import numpy as np

from pkg_resources import resource_filename
from ptsnet.results.workspaces import create_temp_folder
from ptsnet.simulation.constants import TACC_FILE_TEMPLATE

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

def create_tacc_job(
    fpath,
    job_name,
    num_processors,
    allocation,
    run_time,
    processors_per_node = 64,
    queue = 'normal',
    file_args = ''):

    temp_folder_path = create_temp_folder(root=True)
    jobs_path = os.path.join(temp_folder_path, "jobs")
    os.makedirs(jobs_path, exist_ok=True)
    job_path = os.path.join(jobs_path, f'{job_name}.sh')

    job_content = TACC_FILE_TEMPLATE.format(
        job_name = job_name,
        queue = queue,
        num_nodes = int(np.ceil(num_processors/processors_per_node)),
        num_processors = num_processors,
        run_time = str(datetime.timedelta(seconds=run_time*60)),
        allocation = allocation
    )
    job_content += \
        "ml python3\n" + \
        "ml phdf5/1.8.16\n" + \
        f"ibrun python3 {fpath} {file_args}\n"

    with open(job_path, "w") as f:
        f.write(job_content)

def submit_tacc_jobs():
    temp_folder_path = create_temp_folder(root=True)
    submit_path = os.path.join(temp_folder_path, 'submit_jobs.sh')
    jobs_path = os.path.join(temp_folder_path, "jobs")
    fcontent = \
        "#!/bin/bash\n" + \
        f"for f in {jobs_path}/*.sh;\n" + \
        "    do sbatch ${f};\n" + \
        "done\n"
    with open(submit_path, "w") as f:
        f.write(fcontent)
    os.system(f'bash {submit_path}')

def export_time_series(times, data, path):
    '''
        data is a dictionary
    '''
    labels = ['Time']; results = []

    for label in data:
        labels.append(label)
        results.append(data[label])
    header = ','.join(labels)
    np.savetxt(path, list(zip(times, *results)), delimiter=',', header=header, comments='')