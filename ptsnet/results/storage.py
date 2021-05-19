import numpy as np
import pickle
import os
import shutil
import time
import uuid
import json
import h5py

from ptsnet.arrays import Table2D
from ptsnet.simulation.constants import MEM_POOL_POINTS, PIPE_START_RESULTS, PIPE_END_RESULTS, NODE_RESULTS, POINT_PROPERTIES, G, COEFF_TOL
from pkg_resources import resource_filename
from ptsnet.utils.io import get_root_path, walk

class StorageManager:

    def __init__(self, workspace_name, router = None):
        self.root = os.path.join(get_root_path(), 'workspaces')
        self.workspace_path = os.path.join(self.root, workspace_name)
        self._module_path = resource_filename(__name__, '')
        with open( os.path.join(self._module_path,  'metadata.json'), 'r' ) as f:
            self.metadata = json.load(f)
        self.workspace_folders = self.get_workspace_folders()
        self.router = router
        self.data = {}
        self.persistent_files = {}

    def get_workspace_folders(self):
        with open(os.path.join(self._module_path, 'file_structure.json'), 'r') as f:
            fs = json.load(f)
        paths = walk(fs, self.workspace_path)
        tokens = list(map(os.path.basename, paths))
        clean_paths = list(map(os.path.dirname, paths))
        return {int(token) : path for token, path in zip(tokens, clean_paths)}

    def create_workspace_folders(self):
        for folder in self.workspace_folders.values():
            os.makedirs(folder, exist_ok=True)

    def _flush_workspace(self):
        if os.path.isdir(self.workspace_path):
            shutil.rmtree(self.workspace_path)

    def save_data(self, data_label, data, shape = None, comm = None):
        '''
        shape[0] : rows associated with elements
        shape[1] : rows associated with time steps
        '''
        b1 = self.router is None
        b2 = False
        if not b1: b2 = self.router[comm].rank == 0

        idx = self.metadata[data_label]["token"]
        data_path = self.workspace_folders[idx]
        fname = self.metadata[data_label]['fname']
        full_path = os.path.join(data_path, fname)
        file_type = self.metadata[data_label]['ftype']

        if file_type == 'pickle':
            if not (b1 or b2):
                raise SystemError("only processor with rank 0 can store pickle data")
            with open(full_path, 'wb') as f:
                pickle.dump(data, f)
        elif file_type == 'array':
            dtype = self.metadata[data_label]['dtype']
            if b1:
                f = h5py.File(full_path, 'w')
            else:
                f = h5py.File(full_path, 'w', driver='mpio', comm=self.router[comm])
            try:
                data_set = f.create_dataset(data_label, shape, dtype = 'float')
                if not b1:
                    self.router[comm].Barrier()
                if self.router is None:
                    data_set[:] = data
                else:
                    i2 = self.router[comm].scan(data.shape[0])
                    i1 = i2 - data.shape[0]
                    data_set[i1:i2] = data
                f.close()
            except Exception as error:
                f.close()
                raise error

    def load_data(self, data_label):
        d = self.metadata[data_label]
        full_path = os.path.join(self.workspace_folders[d['token']], d['fname'])
        if d['ftype'] == 'array':
            if not full_path in self.persistent_files:
                self.persistent_files[full_path] = h5py.File(full_path, 'r')
            data = self.persistent_files[full_path][data_label]
            return data
        elif d['ftype'] == 'pickle':
            with open(full_path, 'rb') as f:
                return pickle.load(f)

    def close(self):
        if self.persistent_files:
            for key in self.persistent_files:
                self.persistent_files[key].close()

    def exists(self, data_label):
        d = self.metadata[data_label]
        full_path = os.path.join(self.workspace_folders[d['token']], d['fname'])
        return os.path.exists(full_path)