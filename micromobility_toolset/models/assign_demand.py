import numpy as np

from activitysim.core.inject import step, get_injectable

from ..utils.io import (
    load_taz_matrix,
    save_taz_matrix,
    save_node_matrix)


@step()
def assign_demand():

    # initialize configuration data
    trips_settings = get_injectable('trips_settings')
    skims_settings = get_injectable('skims_settings')

    nzones = get_injectable('num_zones')
    bidxs = get_injectable('bike_mode_indices')
    total_demand = np.zeros((nzones, nzones))

    print("\nnon-intrazonal bike trips")
    for segment in trips_settings.get('segments'):

        base_trips = load_taz_matrix(segment)
        bike_trips = np.sum(np.take(base_trips, bidxs, axis=2), 2) * \
                        (np.ones((nzones, nzones)) - np.diag(np.ones(nzones)))

        if segment != 'nhb':
            bike_trips = 0.5 * (bike_trips + np.transpose(bike_trips))

        print(f'{segment}: {round(np.sum(bike_trips), 2)}')

        total_demand = total_demand + bike_trips

    print(f"\ntrip sum: {int(np.sum(total_demand))}")

    print("\nassigning trips to network...")
    base_net = get_injectable('base_network')
    taz_nodes = get_injectable('taz_nodes')
    coef_bike = skims_settings.get('route_varcoef_bike')
    max_cost_bike = skims_settings.get('max_cost_bike')

    base_net.load_attribute_matrix(trips=total_demand,
                                   load_name='bike_vol',
                                   taz_nodes=taz_nodes,
                                   varcoef=coef_bike,
                                   max_cost=max_cost_bike)

    bike_vol = base_net.get_attribute_matrix('bike_vol')

    print(f"\nnetwork sum: {int(np.sum(bike_vol))}")

    print("\nwriting results...")
    save_node_matrix(bike_vol, 'bike_vol.csv')

    print('done.')


if __name__ == '__main__':
    assign_demand()
