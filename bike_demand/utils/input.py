import sqlite3

import numpy as np
import pandas as pd

from .skim import Skim

from activitysim.core.config import (
    setting,
    data_file_path,
    read_model_settings)


def read_taz():

    taz_df = pd.read_csv(data_file_path(setting('taz_table_name') + '.csv'),
                         index_col=setting('taz_table_name'))

    return taz_df.T.to_dict()


def read_matrix(table_name, taz_list):

    trips_settings = read_model_settings('trips.yaml')
    skim = Skim.from_csv(data_file_path(table_name + '.csv'),
                         trips_settings.get('trip_ataz_col'),
                         trips_settings.get('trip_ptaz_col'),
                         mapping=taz_list)

    return skim.to_numpy()
