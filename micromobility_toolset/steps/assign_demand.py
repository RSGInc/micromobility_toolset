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

        total_demand = np.zeros((scenario.num_zones, scenario.num_zones))

        for segment in scenario.trip_settings.get("segments"):

            bike_trips = scenario.load_trip_matrix(segment)
            assert bike_trips.ndim == 2, f"bike trips have shape {bike_trips.shape}"

            if f"{segment}_nhb" in scenario.trip_settings.get("trip_files"):

                nhb_trips = scenario.load_trip_matrix(f"{segment}_nhb")
                bike_trips += nhb_trips

            scenario.logger.info(f"{segment} trips: {round(np.sum(bike_trips), 2)}")

            total_demand = total_demand + bike_trips

        scenario.logger.info(f"trip sum: {int(np.sum(total_demand))}")

        scenario.logger.info("assigning trips to network...")

        scenario.load_network_sums(
            attributes=total_demand,
            cost_attr="bike_cost",  # TODO: parameterize
            load_name="bike_vol",
        )

        link_df = scenario.network.get_link_attributes(["bike_vol", "distance"])
        link_df = link_df[link_df.bike_vol != 0]
        bmt = (link_df["bike_vol"] * link_df["distance"]).sum()
        scenario.logger.info(f"bike miles traveled: {int(bmt)}")

        scenario.logger.info("writing results to bike_vol.csv...")
        link_df["bike_vol"].to_csv(scenario.data_file_path("bike_vol.csv"))
        scenario.logger.info("done.")
