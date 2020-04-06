import os

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
def taz_df():

    taz_df = pd.read_csv(data_file_path(setting('taz_file_name')),
                         index_col=setting('taz_taz_column'))

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
def base_network(trips_settings, network_settings):

    net = Network(network_settings)
    # calculate derived network attributes
    coef_walk = trips_settings.get('route_varcoef_walk')
    coef_bike = trips_settings.get('route_varcoef_bike')

    net.add_derived_network_attributes(coef_walk=coef_walk, coef_bike=coef_bike)

    return net


@inject.injectable(cache=True)
def auto_skim():
    auto_skim_file = setting('auto_skim_file')

    # 3 dimensional matrix with time and distance
    print('reading auto_skim from disk...')
    return read_taz_matrix(data_file_path(auto_skim_file))


@inject.injectable(cache=True)
def bike_skim(trips_settings, base_network, taz_nodes):

    skim_file = setting('bike_skim_file')

    try:
        file_path = data_file_path(skim_file)

        print('reading bike_skim from disk...')
        return read_taz_matrix(file_path)

    except RuntimeError:  # raised if file not found

        print('skimming bike_skim from network...')
        matrix = base_network.get_skim_matrix(taz_nodes,
                                              trips_settings.get('route_varcoef_bike'),
                                              max_cost=trips_settings.get('max_cost_bike'))

        if setting('save_bike_skim'):

            print('saving bike_skim to disk...')
            dist_col = setting('skim_distance_column', default='dist')
            save_taz_matrix(matrix, skim_file, col_names=[dist_col])

        return matrix


@inject.injectable(cache=True)
def walk_skim(trips_settings, base_network, taz_nodes):

    skim_file = setting('walk_skim_file')

    try:
        file_path = data_file_path(skim_file)

        print('reading walk_skim from disk...')
        return read_taz_matrix(file_path)

    except RuntimeError:  # raised if file not found

        print('skimming walk_skim from network...')
        matrix = base_network.get_skim_matrix(taz_nodes,
                                              trips_settings.get('route_varcoef_walk'),
                                              max_cost=trips_settings.get('max_cost_walk'))

        if setting('save_walk_skim'):

            print('saving walk_skim to disk...')
            dist_col = setting('skim_distance_column', default='dist')
            save_taz_matrix(matrix, skim_file, col_names=[dist_col])

        return matrix


@inject.injectable()
def auto_mode_indices(trips_settings):

    all_modes = trips_settings.get('modes')
    auto_modes = trips_settings.get('auto_modes')

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


def read_taz_matrix(file_name):

    trips_settings = inject.get_injectable('trips_settings')
    taz_l = inject.get_injectable('taz_list')

    skim = Skim.from_csv(file_name,
                         trips_settings.get('trip_ataz_col'),
                         trips_settings.get('trip_ptaz_col'),
                         mapping=taz_l)

    return skim.to_numpy()

def load_taz_matrix(segment, base=False):

    trips_settings = inject.get_injectable('trips_settings')
    table_file = trips_settings.get('trip_files').get(segment)

    file_path = data_file_path(table_file)

    # use trip from previous step
    if not base:
        skim = inject.get_injectable(segment, default=None)

        if skim:

            # print('loading cached skim %s' % segment)
            return skim.to_numpy()

        build_file_path = output_file_path(table_file)

        if os.path.exists(build_file_path):
            file_path = build_file_path

    # print('reading %s from %s' % (segment, file_path))
    return read_taz_matrix(file_path)


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
