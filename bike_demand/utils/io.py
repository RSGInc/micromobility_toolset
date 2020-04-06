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
def taz_list():

    return list(inject.get_injectable('taz_df').index)


@inject.injectable(cache=True)
def num_zones():

    return len(inject.get_injectable('taz_list'))


@inject.injectable(cache=True)
def base_network():
    t_settings = inject.get_injectable('trips_settings')
    n_settings = inject.get_injectable('network_settings')

    net = Network(n_settings)
    # calculate derived network attributes
    coef_walk = t_settings.get('route_varcoef_walk')
    coef_bike = t_settings.get('route_varcoef_bike')

    net.add_derived_network_attributes(coef_walk=coef_walk, coef_bike=coef_bike)

    return net


@inject.injectable(cache=True)
def auto_skim():
    auto_skim_file = setting('auto_skim_file')

    return read_taz_matrix(data_file_path(auto_skim_file))


@inject.injectable(cache=True)
def bike_skim():

    t_settings = inject.get_injectable('trips_settings')
    net = inject.get_injectable('base_network')
    tazs = inject.get_injectable('taz_nodes')

    print('skimming bike_skim from network...')
    matrix = net.get_skim_matrix(tazs,
                                 t_settings.get('route_varcoef_bike'),
                                 max_cost=t_settings.get('max_cost_bike'))

    return matrix


@inject.injectable(cache=True)
def walk_skim():

    t_settings = inject.get_injectable('trips_settings')
    net = inject.get_injectable('base_network')
    tazs = inject.get_injectable('taz_nodes')

    print('skimming walk_skim from network...')
    matrix = net.get_skim_matrix(tazs,
                                 t_settings.get('route_varcoef_walk'),
                                 max_cost=t_settings.get('max_cost_walk'))

    return matrix


@inject.injectable()
def auto_mode_indices():

    t_settings = inject.get_injectable('trips_settings')
    all_modes = t_settings.get('modes')
    auto_modes = t_settings.get('auto_modes')

    return [all_modes.index(mode) for mode in auto_modes]


@inject.injectable()
def bike_mode_indices():

    t_settings = inject.get_injectable('trips_settings')
    all_modes = t_settings.get('modes')
    bike_modes = t_settings.get('bike_modes')

    return [all_modes.index(mode) for mode in bike_modes]


@inject.injectable()
def walk_mode_indices():

    t_settings = inject.get_injectable('trips_settings')
    all_modes = t_settings.get('modes')
    walk_modes = t_settings.get('walk_modes')

    return [all_modes.index(mode) for mode in walk_modes]


def load_util_table(segment):

    t_settings = inject.get_injectable('trips_settings')
    table_file = t_settings.get('motorized_util_files').get(segment)

    return read_taz_matrix(data_file_path(table_file))


def read_taz_matrix(file_name):

    t_settings = inject.get_injectable('trips_settings')
    taz_l = inject.get_injectable('taz_list')

    skim = Skim.from_csv(file_name,
                         t_settings.get('trip_ataz_col'),
                         t_settings.get('trip_ptaz_col'),
                         mapping=taz_l)

    return skim.to_numpy()

def load_taz_matrix(segment, base=False):

    t_settings = inject.get_injectable('trips_settings')
    table_file = t_settings.get('trip_files').get(segment)

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

    t_settings = inject.get_injectable('trips_settings')

    if not col_names:
        col_names = t_settings.get('modes')

    skim = Skim(matrix,
                mapping=inject.get_injectable('taz_list'),
                orig_col=t_settings.get('trip_ataz_col'),
                dest_col=t_settings.get('trip_ptaz_col'),
                col_names=col_names)

    t_settings = inject.get_injectable('trips_settings')
    table_file = t_settings.get('trip_files').get(name, name)

    # save the skim for later steps
    inject.add_injectable(name, skim)

    skim.to_csv(output_file_path(table_file))


def save_node_matrix(matrix, name):

    n_settings = inject.get_injectable('network_settings')
    node_list = list(inject.get_injectable('base_network').nodes.keys())

    Skim(matrix,
         mapping=node_list,
         orig_col=n_settings.get('from_name'),
         dest_col=n_settings.get('to_name'),
         col_names=[name]).to_csv(output_file_path(name))
