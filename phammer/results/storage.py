import numpy as np
import zarr, pickle
import os
import shutil
import time
import uuid
import json

from phammer.arrays import ZarrArray
from pkg_resources import resource_filename
from phammer.utils.io import get_root_path, walk

class StorageManager:

    def __init__(self, router = None):
        self.root = get_root_path()
        self.tmp_path = os.path.join(self.root, 'tmp')
        self._module_path = resource_filename(__name__, '')
        with open( os.path.join(self._module_path,  'metadata.json'), 'r' ) as f:
            self.metadata = json.load(f)
        self.tmp_folders = self.get_tmp_folders()
        self.router = router
        self.data = {}

    def get_tmp_folders(self):
        with open(os.path.join(self._module_path, 'file_structure.json'), 'r') as f:
            fs = json.load(f)
        paths = walk(fs, self.tmp_path)
        tokens = list(map(os.path.basename, paths))
        clean_paths = list(map(os.path.dirname, paths))
        return {int(token) : path for token, path in zip(tokens, clean_paths)}

    def create_tmp_folders(self):
        for folder in self.tmp_folders.values():
            os.makedirs(folder, exist_ok=True)

    def flush_tmp(self):
        if os.path.isdir(self.tmp_path):
            shutil.rmtree(self.tmp_path)

    def save_data(self, data_label, data, zarr_shape = None, comm = None):
        '''
        shape[0] : rows associated with elements
        shape[1] : rows associated with time steps
        '''
        b1 = self.router is None
        b2 = False
        if not b1: b2 = self.router[comm].rank == 0

        idx = self.metadata[data_label]["token"]
        data_path = self.tmp_folders[idx]
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
            if dtype == "str":
                dtype = data.dtype
            store = zarr.DirectoryStore(full_path)
            if b1 or b2:
                if len(zarr_shape) > 1:
                    z = zarr.open(store, 'w', shape = zarr_shape, chunks = (1, zarr_shape[1],), dtype = dtype)
                else:
                    z = zarr.open(store, 'w', shape = zarr_shape, chunks = (1,), dtype = dtype)
            if not b1:
                self.router[comm].Barrier()
            z = zarr.open(store, mode = 'r+')
            if self.router is None:
                z[:] = data
            else:
                chunk_size = data.shape[0]
                final_index = self.router[comm].scan(chunk_size)
                initial_index = final_index - chunk_size
                if len(zarr_shape) > 1:
                    z[initial_index:final_index,:] = data
                else:
                    z[initial_index:final_index] = data

    def load_data(self, data_label, indexes = None, labels = None):
        d = self.metadata[data_label]
        full_path = os.path.join(self.tmp_folders[d['token']], d['fname'])
        if d['ftype'] == 'array':
            return ZarrArray(full_path, indexes, labels)
        elif d['ftype'] == 'pickle':
            with open(full_path, 'rb') as f:
                return pickle.load(f)