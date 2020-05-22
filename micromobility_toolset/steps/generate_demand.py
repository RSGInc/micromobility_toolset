import numpy as np
import pandas as pd

from ..model import step


@step()
def generate_demand(*scenarios):
    """
    Generate OD trip tables for scenario using network
    skims and landuse data.

    """


    np.seterr(divide='ignore', invalid='ignore')

    for scenario in scenarios:

        print(f"\nperforming {scenario.name} calculations...")

        bike_avail = (scenario.bike_skim > 0) + np.diag(np.ones(scenario.num_zones))
        buffer_dist = scenario.zone_settings.get('buffer_dist')
        bike_buffer = bike_avail * (1 / (1 + np.exp(4 * (scenario.bike_skim - buffer_dist/2))))

        buffered_zones = pd.DataFrame(index=scenario.zone_list)
        for measure in scenario.zone_settings.get('buffer_cols'):
            zone_col = scenario.zone_df[measure].values
            buffered_zones[measure] = np.sum(zone_col * bike_buffer, axis=0)

        for segment in scenario.trip_settings.get('segments'):

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

            orig_bike_trips = \
                scenario.trip_settings.get('trip_gen_consts')[segment] + \
                np.sum(zone_vals * zone_coefs, axis=1) + \
                np.sum(buffer_vals * buffer_coefs, axis=1)

            orig_bike_trips[orig_bike_trips < 0] = 0
            orig_bike_trips = np.log(orig_bike_trips + 1) * scenario.zone_df[zone_hh_col].values

            dest_cols = scenario.trip_settings.get('dest_choice_coefs')[segment].keys()
            dest_coefs = scenario.trip_settings.get('dest_choice_coefs')[segment].values()
            dest_coefs = np.array(list(dest_coefs))
            dest_vals = scenario.zone_df[dest_cols].values

            dest_size = np.sum(dest_vals * dest_coefs, axis=1)

            intrazonal = \
                np.diag(
                    np.ones(scenario.num_zones) * \
                    scenario.trip_settings.get('bike_intrazonal')[segment])

            gen_dist = \
                scenario.bike_skim + \
                intrazonal + \
                scenario.trip_settings.get('bike_asc')[segment]

            bike_util = np.log(dest_size) + gen_dist
            bike_util = np.exp(bike_util - 999 * (1 - bike_avail))

            # destination-choice fraction
            dc_frac = np.nan_to_num(bike_util / np.sum(bike_util, axis=0))

            # allocate orig trips to destinations
            bike_trips = orig_bike_trips * dc_frac

            print(f"\n{segment} home-based trips")
            print(f'bike: {int(np.sum(bike_trips))}')

            trips = \
                np.zeros((
                    scenario.num_zones,
                    scenario.num_zones,
                    len(scenario.trip_settings.get('modes'))))

            for bike_idx in scenario.bike_mode_indices:
                trips[:, :, bike_idx] = bike_trips
                    #np.nan_to_num(bike_trips / np.sum(bike_trips, axis=2))

            scenario.save_trip_matrix(trips, segment)

            # non-home-based trips
            nhb_factor = scenario.trip_settings.get('nhb_factor').get(segment)
            if nhb_factor:
                nhb_orig_bike_trips = np.sum(bike_trips, axis=1) * nhb_factor
                nhb_bike_trips = nhb_orig_bike_trips * dc_frac
            
                print(f"\n{segment} non-home-based trips")
                print(f'bike: {int(np.sum(nhb_bike_trips))}')

                nhb_trips = \
                    np.zeros((
                        scenario.num_zones,
                        scenario.num_zones,
                        len(scenario.trip_settings.get('modes'))))

                for bike_idx in scenario.bike_mode_indices:
                    nhb_trips[:, :, bike_idx] = nhb_bike_trips
                        #np.nan_to_num(bike_trips / np.sum(bike_trips, axis=2))

                scenario.save_trip_matrix(nhb_trips, f'{segment}_nhb')
