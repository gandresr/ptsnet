import os, shutil
from pkg_resources import resource_filename

RESOURCE_PATH = resource_filename(__name__, 'resources' + os.sep + 'partitions')

def clean_resources():
    for the_file in os.listdir(RESOURCE_PATH):
        file_path = os.path.join(RESOURCE_PATH, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)