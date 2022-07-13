import matplotlib
import matplotlib.pyplot as plt
import pickle
import pdb
from tkinter import font
from ptsnet.graphics.static import plot_knee

matplotlib.rc('ytick', labelsize=28)
with open('tacc_exported_processor_times.pkl', 'rb') as f:
    data = pickle.load(f)
# pp = [22,]# 43, 64]
# tt = [0.0025338773908615115*30,]# 0.001669145731830597, 0.0015908996609687806]
# data['time'] += tt
# data['processor'] += pp
# with open('tacc_exported_processor_times.pkl', 'wb') as f:
#     pickle.dump(data, f)
# exit()
plot_knee('tacc_exported_processor_times.pkl', style='-o', color='#050505')
plot_knee('laptop_exported_processor_times.pkl', style=':o', color='#919090')
plt.legend(['HPC', 'PC'], fontsize=28, frameon=False)
# plt.title('Average Time per Step', fontsize=18)
plt.xlabel('Number of processors', fontsize=24)
plt.ylabel('Time [s]', fontsize=24)
plt.xticks([1, 3, 5, 8, 22], fontsize=28)
plt.tight_layout()
plt.savefig('knee_times.pdf')