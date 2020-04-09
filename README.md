# Micromobility Toolset
This repository contains the software to run incremental demand calculations and benefits
summaries for micro-mobility modes. Test data from the AMBAG bike demand model is provided.

## How to run the example
1. Clone this repository.
2. Run incremental demand:
   ```
   python ambag_bike_model.py --name incremental_demand
   ```
3. Quantify benefits of the changes:
   ```
   python ambag_bike_model.py --name bike_benefits
   ```
4. Assign changed demand counts to the network:
   ```
   python ambag_bike_model.py --name assign_demand
   ```

All steps may be run at once by leaving off the ``--name`` flag:
```
python ambag_bike_model.py
```

The Python script can also generate initial (base) demand using recalculated motorized
utilities.
```
python ambag_bike_model.py --name initial_demand
```

The network skims may also be generated separately from the model steps:
```
python ambag_bike_model.py --name skim_network
```
Note: skims are generated on-the-fly during the main model steps and the
``skim_network`` step does not need to be run beforehand.

### Outputs

Running the model in "initial demand" mode will generate initial demand matrices using the provided
motorized utility tables and input network. This step recalculates the motorized trip utilities for
the input trip matrices to be used in the incremental demand step. Results may be copied to the
base directory for use in incremental_demand.

Running the model in "incremental demand" mode will generate incremental demand matrices. Each
table is indexed by origin and destination zone, and
describes the costs associated with travel between zones:

Running the model in "benefits" mode will quantify the changes in the build directory compared to
the base directory and generate the following outputs:
- **chg_emissions** - i, j, CO2
- **chg_trips** - i, j, da, s2, s3, wt, dt, wk, bk
- **chg_vmt** - i, j, value
- **user_ben** - i, j, minutes of user benefits
Note that benefits will be calculated using pre-"initial demand" trip tables unless the recalculated
trip tables have been copied to the base directory.

All steps will generate network level-of-service matrices (skims) indexed by origin and destination
zone.
- **bike_skim** - i, j, dist
- **walk_skim** - i, j, dist

## Input test data
Input data may be in either CSV or SQLite format. The full test database (SQLite) is available
[here](https://resourcesystemsgroupinc-my.sharepoint.com/:u:/g/personal/ben_stabler_rsginc_com1/EftgpjU25WxKvET6Tmy39tkBRGJZmSeqlyblvzauJ2Iv0w?e=Tfl2nf).

The 25-zone example data (CSV) can be found in ``ambag_example/base/``. Both datasets contain the
following inputs:

### auto skim
Time and distance between zones for calculating change in benefits step.
- **auto_skim** - i, j, time, dist

### network
The network is defined by the following tables (SQLite) or files (CSV):
- **link**:
   - link_type: TWO LANE, MULTILANE, RAMP, FREEWAY, BIKE, PATH, CENTROID C, SHUTTLE
   - length: link length
   - fhwa_fc: FHWA functional class
   - ab_ln: lanes in forward dir
   - ba_ln: lanes in reverse dir
   - area_type: zone area type measure
   - ff_speed: free-flow speed
   - bike_class: 0-4
   - ab_gain: elevation gain in forward dir
   - ba_gain: elevation gain in reverse dir
   - ab_ivt: in-vehicle time in forward dir
   - ba_ivt: in-vehicle time in reverse dir
- **node**
   - node id: node id
   - x_coord: x coordinate
   - y_coord: y coordinate
   - z_coord: z coordinate

### zones
The zones are defined by the following tables:
- **taz**
   - taz: zone number
   - node_id: network node id

### demand
Each table indexed by an origin and destination column, and contains initial zone-to-zone demand by mode. Modes are encoded as drive alone (da), shared ride 2 (s2), shared ride 3+ (sr3), walk transit (wt), drive transit (dt), walk (wk), bike (bk).
- **hbw1trip** - home-based-work 1 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **hbw2trip** - home-based-work 2 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **hbw3trip** - home-based-work 3 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **hbw4trip** - home-based-work 4 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **hscl1trip** - home-based-school 1 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **hscl2trip** - home-based-school 2 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **hscl3trip** - home-based-school 3 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **hscl4trip** - home-based-school 4 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk   
- **hunv1trip** - home-based-univ 1 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk     
- **hunv2trip** - home-based-univ 2 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **hunv3trip** - home-based-univ 3 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **hunv4trip** - home-based-univ 4 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **nhbtrip** - non-home-based trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **nwk1trip** - non-work 1 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **nwk2trip** - non-work 2 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **nwk3trip** - non-work 3 rips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk
- **nwk4trip** - non-work 4 trips, ptaz, ataz, da, s2, s3, wt, dt, wk, bk

Each demand table also has a corresponding motorized utility table to be used in the
"initial demand" step.

### configs
Additional configuration files in the ``configs`` directory
- **settings.yaml** - lists the models and provides taz input configuration
- **network.yaml** - configures network inputs and attributes
- **trips.yaml** - configures demand inputs, mode, and segment coefficients
- **skims.yaml** - configures skim inputs/outputs and network skimming coefficients
