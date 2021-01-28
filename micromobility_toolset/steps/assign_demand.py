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

        scenario.log("performing calculations")
        total_demand = np.zeros((scenario.num_zones, scenario.num_zones))

        for segment in scenario.trip_settings.get('segments'):

            bike_trips = scenario.load_trip_matrix(segment)

            if f'{segment}_nhb' in scenario.trip_settings.get('trip_files'):
                
                nhb_trips = scenario.load_trip_matrix(f'{segment}_nhb')
                bike_trips += nhb_trips
            
            if np.ndim(bike_trips) > 2:
                bike_trips = np.sum(np.take(bike_trips, scenario.bike_mode_indices, axis=2), 2)

            scenario.log(f'{segment} trips: {round(np.sum(bike_trips), 2)}')

            total_demand = total_demand + bike_trips

        scenario.log(f"trip sum: {int(np.sum(total_demand))}")

        scenario.log("assigning trips to network...")

        scenario.network.load_path_attributes(
            paths=scenario.zone_paths,
            attributes=total_demand[scenario.reachable_zones],
            load_name='bike_vol')

        link_df = scenario.network.get_link_attributes(['bike_vol', 'distance'])
        link_df = link_df[link_df.bike_vol != 0]
        bmt = (link_df['bike_vol'] * link_df['distance']).sum()
        scenario.log(f"bike miles traveled: {int(bmt)}")

        scenario.log("writing results to bike_vol.csv...")
        link_df['bike_vol'].to_csv(scenario.data_file_path('bike_vol.csv'))
        scenario.log('done.')
