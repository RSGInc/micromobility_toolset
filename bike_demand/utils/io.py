import pandas as pd

from .skim import Skim
from .network import Network

from activitysim.core import inject
from activitysim.core.config import (
    setting,
    data_file_path,
    read_model_settings)


@inject.injectable(cache=True)
def network_settings():
    return read_model_settings('network.yaml')


@inject.injectable(cache=True)
def trips_settings():
    return read_model_settings('trips.yaml')


@inject.injectable(cache=True)
def taz_df():

    taz_df = pd.read_csv(data_file_path(setting('taz_table_name') + '.csv'),
                         index_col=setting('taz_table_name'))

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
    n_settings = inject.get_injectable('network_settings')
    auto_skim_file = n_settings.get('auto_skim_file')

    return read_matrix(auto_skim_file)

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


def read_matrix(table_name):

    t_settings = inject.get_injectable('trips_settings')
    taz_l = inject.get_injectable('taz_list')

    skim = Skim.from_csv(data_file_path(table_name + '.csv'),
                         t_settings.get('trip_ataz_col'),
                         t_settings.get('trip_ptaz_col'),
                         mapping=taz_l)

    return skim.to_numpy()
