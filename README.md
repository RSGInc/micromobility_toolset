# AMBAG Bike Model
This repository contains all of the code (and links to external data) required to run the bike demand model developed by RSG for AMBAG. 

## Test SQLite DB
The test db is available [here](https://resourcesystemsgroupinc-my.sharepoint.com/:u:/g/personal/ben_stabler_rsginc_com1/EftgpjU25WxKvET6Tmy39tkBRGJZmSeqlyblvzauJ2Iv0w?e=Tfl2nf). It contains the following tables:

### skims
The test db contains the following tables of skims:
- **auto_skim**       
- **bike_skim**
- **walk_skim**

Each table is indexed by origin and destination zone, and describes the costs associated with travel between zones as a function of **time** and **distance**.

### network
The network is defined by the following tables:
- **link**:
   - link_type: TWO LANE, MULTILANE, RAMP, FREEWAY, BIKE, PATH, CENTROID C, SHUTTLE
   - length
   - fhwa_fc
   - ab_ln: lanes in forward dir
   - ba_ln: lanes in reverse dir
   - area_type
   - ff_speed: free-flow speed
   - bike_class: 0-4
   - ab_gain: elevation gain in forward dir
   - ba_gain: elevation gain in reverse dir
   - ab_ivt: in-vehicle time in forward dir
   - ba_ivt: in-vehicle time in reverse dir
- **node**
   - node id
   - x_coord
   - y_coord
   - z_coord
- **linkpoint**
   - id
   - pointno
   - x_coord
   - y_coord
   - z_coord
   
### zones
- **taz**
   - taz       
   - area      
   - co        
   - node_id   
   - households
   - inc_1     
   - inc_2     
   - inc_3     
   - inc_4     
   - age24under
   - age25to44 
   - age45to64 
   - age65plus 
   - auto_0    
   - auto_1    
   - auto_2    
   - auto_3    
   - farm      
   - indu      
   - cons      
   - retl      
   - serv      
   - govt      
   - emp       
   - k12_enroll
   - unv_enroll
- **tazpoly**
   - taz    
   - polyno 
   - pointno
   - xcoord 
   - ycoord 
   
### Demand
The test db contains the following tables of trips (demand):
- **hbw1trip**
- **hbw2trip**       
- **hbw3trip**       
- **hbw4trip**       
- **hscl1trip**      
- **hscl2trip**      
- **hscl3trip**
- **hscl4trip**      
- **hunv1trip**      
- **hunv2trip**      
- **hunv3trip**      
- **hunv4trip**
- **nhbtrip**       
- **nwk1trip**      
- **nwk2trip**      
- **nwk3trip**
- **nwk4trip**

Each table ndexed by an origin and destination column, and contains initial zone-to-zone demand by mode. Modes are encoded as the following two-character strings: "da", "s2", "s3", "wt", "dt", "wk", "bk".
 
### Outputs
Running the model in "benefits" mode will generate the following three tables of outputs:
- **chg_emissions**     
- **chg_trips**         
- **chg_vmt**

Each table is indexed by origin and destination zone.

### Misc:
- **project_info**
- **user_ben**  

## How To
1. Clone this repository.
2. Download test SQLite db  and copy it into this main directory of this repository.
3. Navigate to the `scripts/` directory.
4. Run the Python script to generate incremental demand:
   ```
   python ambag_bike_model_python.py --type incremental_demand --base ../new_path_coef.db --build ../new_path_coef.db
   ```
5. Run the Python script to quantify benefits of the changes:
   ```
   python ambag_bike_model_python.py --type benefits --base ../new_path_coef.db --build ../new_path_coef.db
   ```
   
