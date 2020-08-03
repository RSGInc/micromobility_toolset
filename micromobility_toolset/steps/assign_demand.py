import numpy as np

from ..model import step


@step()
def assign_demand(*scenarios):
    """
    This step adds zone-to-zone bike trips to the network, which
    fills in the bike volume for all the links on each route. These links
    and their bike volumes are written to a file.
    """

    for scenario in scenarios:

        print(f"\nperforming {scenario.name} calculations")
        total_demand = np.zeros((scenario.num_zones, scenario.num_zones))

        for segment in scenario.trip_settings.get('segments'):

            bike_trips = scenario.load_trip_matrix(segment)

            if f'{segment}_nhb' in scenario.trip_settings.get('trip_files'):
                
                nhb_trips = scenario.load_trip_matrix(f'{segment}_nhb')
                bike_trips += nhb_trips
            
            if np.ndim(bike_trips) > 2:
                bike_trips = np.sum(np.take(bike_trips, scenario.bike_mode_indices, axis=2), 2)

            print(f'{segment}: {round(np.sum(bike_trips), 2)}')

            total_demand = total_demand + bike_trips

        print(f"\ntrip sum: {int(np.sum(total_demand))}")

        print("\nassigning trips to network...")

        scenario.network.load_attribute_matrix(
            matrix=total_demand,
            load_name='bike_vol',
            centroid_ids=scenario.zone_nodes,
            varcoef=scenario.network_settings.get('route_varcoef_bike'),
            max_cost=scenario.network_settings.get('max_cost_bike'))

        bike_vol = scenario.network.link_df['bike_vol']
        bmt = bike_vol * scenario.network.link_df['distance']
        print(f"\nbike miles traveled: {int(bmt.sum())}")

        print("\nwriting results...")
        bike_vol.to_csv(scenario.data_file_path('bike_vol.csv'))
        print('done.')
