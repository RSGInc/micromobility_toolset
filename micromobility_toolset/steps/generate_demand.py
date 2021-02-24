import numpy as np
import pandas as pd

from ..model import step


@step()
def generate_demand(*scenarios):
    """
    Generate Production-Attraction trip tables for scenario using network
    skims and landuse data.
    """

    np.seterr(divide='ignore', invalid='ignore')

    for scenario in scenarios:

        dest_avail = (scenario.distance_skim > 0)
        buffer_dist = scenario.zone_settings.get('buffer_dist')
        zone_buffer = dest_avail * (1 / (1 + np.exp(4 * (scenario.distance_skim - buffer_dist/2))))

        # initialize dataframes
        buffered_zones = pd.DataFrame(index=scenario.zone_list)
        trip_gen_df = pd.DataFrame(index=scenario.zone_list)
        dest_size_df = pd.DataFrame(index=scenario.zone_list)

        for measure in scenario.zone_settings.get('buffer_cols'):
            zone_col = scenario.zone_df[measure].values
            buffered_zones[measure] = np.sum(zone_col * zone_buffer, axis=0)

        for segment in scenario.trip_settings.get('segments'):


            orig_trips = create_trips(scenario, segment, buffered_zones)

            # save segment trips to production df
            trip_gen_df[segment] = orig_trips

            dest_cols = scenario.trip_settings.get('dest_choice_coefs')[segment].keys()
            dest_coefs = scenario.trip_settings.get('dest_choice_coefs')[segment].values()
            dest_coefs = np.array(list(dest_coefs))
            dest_vals = scenario.zone_df[dest_cols].values

            dest_size = np.sum(dest_vals * dest_coefs, axis=1)
            dest_size[dest_size < 0] = 0

            # save segment values to attraction df
            dest_size_df[segment] = dest_size

            distribute_trips(scenario, segment, orig_trips, dest_size, dest_avail)


        # finally, save intermediate calculations to disk
        buffered_zones.round(4).to_csv(scenario.data_file_path('buffered_zones.csv'))
        trip_gen_df.round(4).to_csv(scenario.data_file_path('zone_production_size.csv'))
        dest_size_df.round(4).to_csv(scenario.data_file_path('zone_attraction_size.csv'))


def create_trips(scenario, segment, buffered_zones):
    # origin zone trips
    zone_hh_col = scenario.trip_settings.get('hh_col')
    zone_cols = scenario.trip_settings.get('trip_gen_zone_coefs')[segment].keys()
    zone_coefs = scenario.trip_settings.get('trip_gen_zone_coefs')[segment].values()
    zone_coefs = np.array(list(zone_coefs))
    zone_vals = scenario.zone_df[zone_cols].values

    buffer_cols = scenario.trip_settings.get('trip_gen_buffer_coefs')[segment].keys()
    buffer_coefs = scenario.trip_settings.get('trip_gen_buffer_coefs')[segment].values()
    buffer_coefs = np.array(list(buffer_coefs))
    buffer_vals = buffered_zones[buffer_cols].values

    orig_trips = \
        scenario.trip_settings.get('trip_gen_consts')[segment] + \
        np.sum(zone_vals * zone_coefs, axis=1) + \
        np.sum(buffer_vals * buffer_coefs, axis=1)

    orig_trips[orig_trips < 0] = 0

    # multiply by households to get total trip counts
    orig_trips = orig_trips * scenario.zone_df[zone_hh_col].values

    return orig_trips


def distribute_trips(scenario, segment, orig_trips, dest_size, dest_avail):

    intrazonal = \
        np.diag(
            np.ones(scenario.num_zones) * \
            scenario.trip_settings.get('bike_intrazonal')[segment])

    # TODO: parameterize
    gen_cost = \
        scenario.skims.get_core('bike_cost') + \
        intrazonal + \
        scenario.trip_settings.get('bike_asc')[segment]

    bike_util = np.log(dest_size + 1) + gen_cost
    bike_util = np.exp(bike_util - 999 * (1 - dest_avail))

    # destination-choice fraction
    dc_frac = np.nan_to_num(bike_util / np.sum(bike_util, axis=1).reshape(-1,1))
    # print(np.sum(dc_frac, axis=1))  # should be all ones

    # allocate orig trips to destinations
    bike_trips = orig_trips.reshape(-1,1) * dc_frac

    scenario.logger.info(f'{segment} home-based trips: {int(np.sum(bike_trips))}')

    scenario.save_trip_matrix(bike_trips, segment)

    # non-home-based trips
    nhb_factor = scenario.trip_settings.get('nhb_factor').get(segment)
    if nhb_factor:
        nhb_orig_trips = np.sum(bike_trips, axis=1) * nhb_factor
        nhb_bike_trips = nhb_orig_trips.reshape(-1,1) * dc_frac

        scenario.logger.info(f'{segment} non-home-based trips: {int(np.sum(nhb_bike_trips))}')

        scenario.save_trip_matrix(nhb_bike_trips, f'{segment}_nhb')
