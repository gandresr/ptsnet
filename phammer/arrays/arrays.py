from vaex.dataframe import DataFrame
import numpy as np

def define_data_table(elements, props, dtype=np.int):
    return DataFrame(
        np.full((len(elements), len(props)), np.nan, dtype=dtype),
        index = elements,
        columns = props)

def define_results_table(elements, time_steps, dtype=np.float64):
    return DataFrame(
        np.zeros((len(elements), time_steps), dtype=dtype),
        index = elements)