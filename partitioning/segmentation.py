from wntr_graph import *

model = "models/UTnet_v2_April2016"
G = get_segmented_graph(model, 0.004)
write_metis_file(model, 0.004)