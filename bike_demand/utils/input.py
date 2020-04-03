import sqlite3

import numpy as np
import pandas as pd

from .skim import Skim

from activitysim.core import inject
from activitysim.core.config import (
    setting,
    data_file_path,
    read_model_settings)


@inject.injectable(cache=True)
def trips_settings():
    return read_model_settings('trips.yaml')

@inject.injectable(cache=True)
def taz_data():

    taz_df = pd.read_csv(data_file_path(setting('taz_table_name') + '.csv'),
                         index_col=setting('taz_table_name'))

    return taz_df.T.to_dict()


def read_matrix(table_name):

    taz_list = list(inject.get_injectable('taz_data').keys())
    trips_settings = inject.get_injectable('trips_settings')
    skim = Skim.from_csv(data_file_path(table_name + '.csv'),
                         trips_settings.get('trip_ataz_col'),
                         trips_settings.get('trip_ptaz_col'),
                         mapping=taz_list)

    return skim.to_numpy()
