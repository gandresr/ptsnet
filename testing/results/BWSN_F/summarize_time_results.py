import os

import numpy as np
import pickle as pk
from collections import defaultdict as ddict

folders = os.listdir()

for folder in folders:
    print(folder)
    if '.' in folder: continue
    files = os.listdir(folder)
    full_data = ddict(list)
    for file in files:
        if '.dat' in file: continue
        print(file)
        with open(folder + '/' + file, 'rb') as f:
            data = pk.load(f)
        j = data['sim_times'].jobs
        for k in j.keys():
            full_data[k] += j[k].durations
    summary = {}
    for k in full_data.keys():
        median = np.median(full_data[k])
        mean = np.mean(full_data[k])
        q1 = np.percentile(full_data[k], 25)
        q3 = np.percentile(full_data[k], 75)
        mmax = np.max(full_data[k])
        mmin = np.min(full_data[k])
        summary[k] = [mmin, q1, median, q3, mmax, mean]
    with open(folder + '/summary.dat', 'wb') as fs:
        pk.dump(summary, fs)
