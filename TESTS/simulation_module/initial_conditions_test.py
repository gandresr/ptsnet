import importlib.util

pmoc_path = "/home/watsup/Documents/Github/hammer-net/partitioning/pmoc.py"
spec = importlib.util.spec_from_file_location("pmoc", pmoc_path)
pmoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pmoc)
Simulation = pmoc.Simulation