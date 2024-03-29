import numpy as np
import pandas as pd

from ..model import step


@step()
def generate_demand(*scenarios):
    """
    Generate Production-Attraction trip tables for scenario using network
    skims and landuse data.
    """

    np.seterr(divide="ignore", invalid="ignore")

    for scenario in scenarios:

        buffer_dist = scenario.zone_settings.get("buffer_dist")
        zone_buffer = 1 / (1 + np.exp(4 * (scenario.distance_skim - buffer_dist / 2)))
        zone_buffer[scenario.distance_skim == 0] = 0
        np.fill_diagonal(zone_buffer, 1.0)

        # initialize dataframes
        buffered_zones = pd.DataFrame(index=scenario.zone_list)
        trip_gen_df = pd.DataFrame(index=scenario.zone_list)
        dest_size_df = pd.DataFrame(index=scenario.zone_list)

        dup_node_idxs = scenario.duplicate_nodes()
        zone_array = np.array(scenario.zone_list)
        from_zones = zone_array[dup_node_idxs[0]]
        to_zones = zone_array[dup_node_idxs[1]]
        duplicates_df = pd.DataFrame(index=[from_zones, to_zones])

        for measure in scenario.zone_settings.get("buffer_cols"):
            zone_col = scenario.zone_df[measure].values
            buffered_zones[measure] = np.sum(zone_col * zone_buffer, axis=1)

        for segment in scenario.trip_settings.get("segments"):

            orig_trips = create_trips(scenario, segment, buffered_zones)

            # save segment trips to production df
            trip_gen_df[segment] = orig_trips

            dest_size = calc_dest_size(scenario, segment, dest_size_df)
            bike_trips = distribute_trips(scenario, segment, orig_trips, dest_size)

            duplicates_df[segment] = bike_trips[dup_node_idxs]

            scenario.save_trip_matrix(bike_trips, segment)

        # finally, save intermediate calculations to disk
        buffered_zones.round(4).to_csv(scenario.data_file_path("buffered_zones.csv"))
        trip_gen_df.round(4).to_csv(scenario.data_file_path("zone_production_size.csv"))
        dest_size_df.round(4).to_csv(
            scenario.data_file_path("zone_attraction_size.csv")
        )
        duplicates_df.round(4).to_csv(scenario.data_file_path("non_network_trips.csv"))

        scenario.logger.info(
            f"non-network trip count: {int(duplicates_df.sum().sum())}"
        )


def create_trips(scenario, segment, buffered_zones):
    # origin zone trips

    sum_factor = scenario.trip_settings.get("trip_gen_sum_factor").get(segment)
    if sum_factor:

        reference_trips = scenario.load_trip_matrix(sum_factor["segment"])

        # home-based trip destinations become nhb origins
        orig_trips = (
            np.sum(reference_trips, axis=sum_factor["axis"]) * sum_factor["coef"]
        )

        return orig_trips

    zone_hh_col = scenario.zone_settings.get("zone_hh_col")
    zone_cols = scenario.trip_settings.get("trip_gen_zone_coefs")[segment].keys()
    zone_coefs = scenario.trip_settings.get("trip_gen_zone_coefs")[segment].values()
    zone_coefs = np.array(list(zone_coefs))
    zone_vals = scenario.zone_df[zone_cols].values

    buffer_cols = scenario.trip_settings.get("trip_gen_buffer_coefs")[segment].keys()
    buffer_coefs = scenario.trip_settings.get("trip_gen_buffer_coefs")[segment].values()
    buffer_coefs = np.array(list(buffer_coefs))
    buffer_vals = buffered_zones[buffer_cols].values

    orig_trips = (
        scenario.trip_settings.get("trip_gen_consts")[segment]
        + np.sum(zone_vals * zone_coefs, axis=1)
        + np.sum(buffer_vals * buffer_coefs, axis=1)
    )

    orig_trips[orig_trips < 0] = 0

    # multiply by households to get total trip counts
    orig_trips = orig_trips * scenario.zone_df[zone_hh_col].values

    return orig_trips


def calc_dest_size(scenario, segment, dest_size_df):

    reuse = scenario.trip_settings.get("reuse_dest_size").get(segment)
    if reuse:

        dest_size = dest_size_df[reuse].values

        return dest_size

    dest_cols = scenario.trip_settings.get("dest_choice_zone_coefs")[segment].keys()
    dest_coefs = scenario.trip_settings.get("dest_choice_zone_coefs")[segment].values()
    dest_coefs = np.array(list(dest_coefs))
    dest_vals = scenario.zone_df[dest_cols].values

    dest_size = np.sum(dest_vals * dest_coefs, axis=1)
    dest_size[dest_size < 0] = 0

    # save segment values to attraction df
    dest_size_df[segment] = dest_size

    return dest_size


def distribute_trips(scenario, segment, orig_trips, dest_size):

    max_dist = scenario.trip_settings.get("trip_max_dist").get(segment, np.inf)
    min_dist = scenario.trip_settings.get("trip_min_dist").get(segment, 0)

    dest_avail = (scenario.distance_skim > min_dist) * (
        scenario.distance_skim < max_dist
    )

    # include intrazonal trips
    if min_dist == 0:
        np.fill_diagonal(dest_avail, True)

    cost_attr = scenario.trip_settings.get("trip_cost_attr")[segment]
    gen_cost = (
        - scenario.skims.get_core(cost_attr).astype(np.float32)
        + np.diag(np.full(scenario.num_zones, scenario.trip_settings.get("bike_intrazonal")[segment])).astype(np.float32)  # intrazonals
        + scenario.trip_settings.get("bike_asc")[segment]
    ).astype(np.float32)

    bike_util = (np.log(dest_size) + gen_cost).astype(np.float32)
    bike_util = np.exp(bike_util - 999 * (1 - dest_avail))

    # destination-choice fraction
    dc_frac = np.nan_to_num(bike_util / np.sum(bike_util, axis=1).reshape(-1, 1)).astype(np.float32)
    #print(np.sum(dc_frac, axis=1))  # should be all ones

    # allocate orig trips to destinations
    bike_trips = (orig_trips.reshape(-1, 1) * dc_frac).astype(np.float32)

    scenario.logger.info(f"{segment} trips: {int(np.sum(bike_trips))}")

    return bike_trips
