import matplotlib.pyplot as plt
from ptsnet.graphics.static import plot_knee

plot_knee('tacc_exported_processor_times.pkl')
plot_knee('laptop_exported_processor_times.pkl')
plt.yscale('log')
plt.show()