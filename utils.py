import os
from typing import List

import pandas as pd


class Dataset:
    def __init__(self, dirpath: str, condition_dir_name: str = "condition") -> None:
        condition_dirpath = os.path.join(dirpath, condition_dir_name)
        control_dirpath = os.path.join(dirpath, "control")

        self.condition: List[pd.DataFrame] = \
            [pd.read_csv(os.path.join(condition_dirpath, file)) for file in os.listdir(condition_dirpath)]

        self.control: List[pd.DataFrame] = \
            [pd.read_csv(os.path.join(control_dirpath, file)) for file in os.listdir(control_dirpath)]
