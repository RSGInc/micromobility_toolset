# AMBAG Bike Model
This repository contains the software and input test data required to run the AMBAG bike demand model.

## How To
1. Clone this repository.
2. Run the Python script to generate initial demand:
   ```
   python ambag_bike_model.py --name initial_demand
   ```
3. Run the Python script to generate incremental demand:
   ```
   python ambag_bike_model.py --name incremental_demand
   ```
4. Run the Python script to quantify benefits of the changes:
   ```
   python ambag_bike_model.py --name bike_benefits
   ```
5. Run the Python script to assign demand counts to the changes:
  ```
  python ambag_bike_model.py --name assign_demand
  ```

All steps may be run at once by leaving off the ``--name`` flag:
```
python ambag_bike_model.py
```

The network skims may also be generated separately from the model steps:
```
python ambag_bike_model.py --name skim_network
```
Note: skims are generated on-the-fly during the main model steps and the
``skim_network`` step does not need to be run beforehand.

### Outputs

Running the model in "incremental demand" mode will generate network level-of-service matrices (skims) and incremental demand matrices.  Each table is indexed by origin and destination zone, and describes the costs associated with travel between zones:
- **auto_skim** - i, j, time, dist
- **bike_skim** - i, j, dist
- **walk_skim** - i, j, dist

Running the model in "benefits" mode will generate the following three tables of outputs:
- **chg_emissions** - i, j, CO2
- **chg_trips** - i, j, da, s2, s3, wt, dt, wk, bk
- **chg_vmt** - i, j, value
- **user_ben** - i, j, minutes of user benefits

## Input test data
The full test database is available [here](https://resourcesystemsgroupinc-my.sharepoint.com/:u:/g/personal/ben_stabler_rsginc_com1/EftgpjU25WxKvET6Tmy39tkBRGJZmSeqlyblvzauJ2Iv0w?e=Tfl2nf).

The 25-zone example can be found in ``ambag_example/base/``. Both datasets contain the following inputs:

### network
The network is defined by the following tables:
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
   - area: area
   - co: county
   - node_id: network node id
   - households: number of households
   - inc_1: households by income class
   - inc_2: households by income class
   - inc_3: households by income class
   - inc_4: households by income class
   - age24under: households by age class
   - age25to44: households by age class
   - age45to64: households by age class
   - age65plus: households by age class
   - auto_0: households by auto class
   - auto_1: households by auto class
   - auto_2: households by auto class
   - auto_3: households by auto class
   - farm: farm employment
   - indu: industrial employment
   - cons: construction employment
   - retl: retail employment
   - serv: service employment
   - govt: government employment
   - emp: total employment
   - k12_enroll: K-12 enrollment
   - unv_enroll: university enrollment
- **tazpoly**
   - taz: zone number
   - polyno: zone polygon id
   - pointno: zone point id
   - xcoord: zone point x coordinate
   - ycoord: zone point y coordinate

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
