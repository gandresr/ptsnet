import os
import datetime
import numpy as np

from ptsnet.simulation.constants import TACC_FILE_TEMPLATE
from ptsnet.results.workspaces import get_tmp_folder

def create_tacc_job(
    fpath,
    job_name,
    num_processors,
    allocation,
    run_time,
    processors_per_node = 64,
    queue = 'normal',
    file_args = ''):

    temp_folder_path = get_tmp_folder()
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
    temp_folder_path = get_tmp_folder()
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