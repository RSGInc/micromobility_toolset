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

        cost_map = scenario.trip_settings.get("trip_cost_attr")
        cost_attrs = list(set(list(cost_map.values())))

        for cost_attr in cost_attrs:

            total_demand = np.zeros((scenario.num_zones, scenario.num_zones))

            for segment in scenario.trip_settings.get("segments"):

                if cost_attr != cost_map[segment]:
                    continue

                scenario.logger.debug(f"adding {segment} demand to {cost_attr} sums")
                bike_trips = scenario.load_trip_matrix(segment)

                if f"{segment}_nhb" in scenario.trip_settings.get("trip_files"):

                    nhb_trips = scenario.load_trip_matrix(f"{segment}_nhb")
                    bike_trips += nhb_trips

                scenario.logger.info(f"{segment} trips: {round(np.sum(bike_trips), 2)}")

                total_demand = total_demand + bike_trips

            scenario.logger.info(f"{cost_attr} trip sum: {int(np.sum(total_demand))}")

            scenario.logger.info(f"assigning {cost_attr} trips to network...")

            scenario.load_network_sums(
                attributes=total_demand,
                cost_attr=cost_attr,
                load_name=f"{cost_attr}_vol",
            )

        vol_cols = [f"{attr}_vol" for attr in cost_attrs]
        link_df = scenario.network.get_link_attributes(vol_cols + ["distance"])
        link_df["bike_vol"] = link_df[vol_cols].sum(axis=1)
        link_df = link_df[link_df.bike_vol != 0]
        bmt = (link_df["bike_vol"] * link_df["distance"]).sum()
        scenario.logger.info(f"bike miles traveled: {int(bmt)}")

        scenario.logger.info("writing results to bike_vol.csv...")
        link_df.bike_vol.to_csv(scenario.data_file_path("bike_vol.csv"))
        scenario.logger.info("done.")
