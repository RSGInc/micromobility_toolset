import os
import sqlite3

import pandas as pd

from activitysim.core import inject
from activitysim.core.config import (
    setting,
    data_file_path,
    output_file_path,
    read_model_settings)

from .skim import Skim
from .network import Network


@inject.injectable(cache=True)
def network_settings():

    return read_model_settings('network.yaml')


@inject.injectable(cache=True)
def trips_settings():

    return read_model_settings('trips.yaml')


@inject.injectable(cache=True)
def skims_settings():

    return read_model_settings('skims.yaml')


@inject.injectable(cache=True)
def taz_df():

    file_path = data_file_path(setting('taz_file_name'))

    if file_path.endswith('.csv'):
        taz_df = pd.read_csv(file_path, index_col=setting('taz_taz_column'))

    elif file_path.endswith('.db'):
        db_connection = sqlite3.connect(file_path)

        taz_df = pd.read_sql(f"select * from {setting('taz_table_name')}",
                             db_connection,
                             index_col=setting('taz_taz_column'))

        db_connection.close()

    else:
        raise TypeError(f"cannot read TAZ filetype {os.path.basename(file_path)}")

    print('loaded %s zones' % str(taz_df.shape[0]))
    return taz_df


# TODO: change network to use df instead of dict
@inject.injectable(cache=True)
def taz_nodes():

    df = inject.get_injectable('taz_df')
    nodes = df[setting('taz_node_column')]

    return nodes.T.to_dict()


@inject.injectable(cache=True)
def taz_list(taz_df):

    return list(taz_df.index)


@inject.injectable(cache=True)
def num_zones(taz_list):

    return len(taz_list)


@inject.injectable(cache=True)
def base_network(skims_settings, network_settings):

    net = Network(network_settings)
    # calculate derived network attributes
    coef_walk = skims_settings.get('route_varcoef_walk')
    coef_bike = skims_settings.get('route_varcoef_bike')

    net.add_derived_network_attributes(coef_walk=coef_walk, coef_bike=coef_bike)

    return net


@inject.injectable(cache=True)
def auto_skim(skims_settings, taz_list):
    auto_skim_file = skims_settings.get('auto_skim_file')
    ataz_col = skims_settings.get('skim_ataz_col')
    ptaz_col = skims_settings.get('skim_ptaz_col')

    # 3 dimensional matrix with time and distance
    print('reading auto_skim from disk...')
    skim = Skim.from_csv(data_file_path(auto_skim_file),
                         ataz_col, ptaz_col,
                         mapping=taz_list)

    return skim


def load_skim(mode):

    skim = inject.get_injectable(mode + '_skim', default=None)

    if skim:

        # print('loading cached skim %s' % segment)
        return skim.to_numpy()

    taz_list = inject.get_injectable('taz_list')
    skims_settings = inject.get_injectable('skims_settings')
    skim_file = skims_settings.get(mode + '_skim_file')
    ataz_col = skims_settings.get('skim_ataz_col')
    ptaz_col = skims_settings.get('skim_ptaz_col')
    file_path = output_file_path(skim_file)

    if os.path.exists(file_path):

        print('loading %s skim from %s' % (mode, inject.get_injectable('output_dir')))
        taz_l = inject.get_injectable('taz_list')

        skim = Skim.from_csv(file_path, ataz_col, ptaz_col, mapping=taz_list)
        return skim.to_numpy()

    net = inject.get_injectable('base_network')
    taz_nodes = inject.get_injectable('taz_nodes')

    print('skimming %s skim from network...' % mode)
    matrix = net.get_skim_matrix(taz_nodes,
                                 skims_settings.get('route_varcoef_' + mode),
                                 max_cost=skims_settings.get('max_cost_' + mode))

    skim = Skim(matrix,
                mapping=taz_list,
                orig_col=ataz_col,
                dest_col=ptaz_col,
                col_names=[skims_settings.get('skim_distance_col', 'distance')])

    inject.add_injectable(mode + '_skim', skim)

    if skims_settings.get('save_%s_skim' % mode, False):

        print('saving %s skim to disk...' % mode)
        skim.to_csv(file_path)

    return matrix


@inject.injectable()
def motorized_mode_indices(trips_settings):

    all_modes = trips_settings.get('modes')
    auto_modes = trips_settings.get('motorized_modes')

    return [all_modes.index(mode) for mode in auto_modes]


@inject.injectable()
def bike_mode_indices(trips_settings):

    all_modes = trips_settings.get('modes')
    bike_modes = trips_settings.get('bike_modes')

    return [all_modes.index(mode) for mode in bike_modes]


@inject.injectable()
def walk_mode_indices(trips_settings):

    all_modes = trips_settings.get('modes')
    walk_modes = trips_settings.get('walk_modes')

    return [all_modes.index(mode) for mode in walk_modes]


def load_util_table(segment):

    trips_settings = inject.get_injectable('trips_settings')
    table_file = trips_settings.get('motorized_util_files').get(segment)

    return read_taz_matrix(data_file_path(table_file))


def read_taz_matrix(file_name, table_name=None):

    trips_settings = inject.get_injectable('trips_settings')
    taz_l = inject.get_injectable('taz_list')
    ataz_col = trips_settings.get('trip_ataz_col')
    ptaz_col = trips_settings.get('trip_ptaz_col')

    if file_name.endswith('.csv'):
        skim = Skim.from_csv(file_name, ataz_col, ptaz_col, mapping=taz_l)

    elif file_name.endswith('.db'):
        skim = Skim.from_sqlite(file_name, table_name,
                                ataz_col, ptaz_col,
                                mapping=taz_l)

    else:
        raise TypeError(f"cannot read matrix from filetype {os.path.basename(file_name)}")

    return skim.to_numpy()

def load_taz_matrix(segment, base=False):

    trips_settings = inject.get_injectable('trips_settings')
    csv_file = trips_settings.get('trip_files').get(segment)

    # use CSVs for input if no sqlite db provided
    table_file = trips_settings.get('input_sqlite_db', csv_file)
    table_name = trips_settings.get('trip_tables', {}).get(segment)
    file_path = data_file_path(table_file)

    if not base:

        # use trip from previous step
        skim = inject.get_injectable(segment, default=None)

        if skim:

            # print('loading cached skim %s' % segment)
            return skim.to_numpy()

        build_file_path = output_file_path(table_file)

        if os.path.exists(build_file_path):
            file_path = build_file_path

    return read_taz_matrix(file_path, table_name=table_name)



def save_taz_matrix(matrix, name, col_names=None):

    trips_settings = inject.get_injectable('trips_settings')

    if not col_names:
        col_names = trips_settings.get('modes')

    skim = Skim(matrix,
                mapping=inject.get_injectable('taz_list'),
                orig_col=trips_settings.get('trip_ataz_col'),
                dest_col=trips_settings.get('trip_ptaz_col'),
                col_names=col_names)

    trips_settings = inject.get_injectable('trips_settings')
    table_file = trips_settings.get('trip_files').get(name, name)

    # save the skim for later steps
    inject.add_injectable(name, skim)

    skim.to_csv(output_file_path(table_file))


def save_node_matrix(matrix, name):

    network_settings = inject.get_injectable('network_settings')
    node_list = list(inject.get_injectable('base_network').nodes.keys())

    Skim(matrix,
         mapping=node_list,
         orig_col=network_settings.get('from_name'),
         dest_col=network_settings.get('to_name'),
         col_names=[name]).to_csv(output_file_path(name))
