# Micromobility Toolset
Welcome to the Micromobility Toolset, built by RSG Inc. in collaboration with [Wasatch Front Regional Council](https://github.com/WFRCAnalytics). This repository contains the software to run several trip generation and network assignment calculations for micro-mobility modes. The latest model specification doc can be found in the root directory of this repository. This document describes the technical design of the model and contains details about calculations, methods, and configuration.

## Quick Start

To quickly get up and running, first ensure Git and Python 3.6 or greater are installed on your computer. Then clone the repository with

```
git clone https://github.com/RSGInc/micromobility_toolset.git
```

Navigate to the project folder (`cd micromobility_toolset`) and install the program with

```
python setup.py install
```

Run the example model with

```
cd wfrc_example
python utah_bike_demand_model.py
```

Example inputs and configuration files can be found in the `Model_Inputs/` and `Model_Configs/` subdirectories of `wfrc_example/`. Run `python utah_bike_demand_model.py -h` for more information on run options.

Alternatively, the package can be installed with

```
pip install git+https://github.com/RSGInc/micromobility_toolset.git
```

## Input data
Input data for the full Salt Lake regional model can be found at [WFRC's official repository](https://github.com/WFRCAnalytics/utah_bike_demand_model). A smaller version that mirrors the schema and configuration of the full model can be found included in this repository. A detailed list of the input data attributes can be found [here](https://github.com/WFRCAnalytics/utah_bike_demand_model/tree/master/Model_Inputs).

### network
The network is defined by a link file and a node file containing traffic network attributes.

- **link.csv**:
- **node.csv**

### zones
The zones are defined by a single zone table. Additional attributes are necessary for the `generate_demand` step.
- **zone.csv**
   - taz: taz number
   - node_id: network node id

## Configuration
Configuration files in the ``configs`` directory
- **zone.yaml** - provides zone input configuration
- **network.yaml** - configures network inputs and attributes
- **trips.yaml** - configures demand inputs, mode, and segment coefficients

## Outputs

The program will create the traffic network using the link and node inputs provided and save the graph to the output directory. It will also create the skims (zone-to-zone cost matrices) for each of the link weights specified in the network configuration file. Costs between every zone in the zone file will be calculated, up to a maximum configured value. Each model step will then generate additional outputs using the network and configuration parameters.

## Model Steps

When first running the model, the program will first build a graph representation of the traffic network using the given link and node input files. Each model will then use a combination of the provided land-use and network data, along with the configuration coefficients, to produce various outputs. Some of these outputs, like the trip tables from `Generate Demand`, will be used for subsequent model steps. Most steps require a zone level OD matrix to perform their calculations -- if this matrix is not provided as an input, it will be skimmed from the network and saved to the output directory. Subsequent steps will reuse this file in favor of recalculating the OD matrix.

### Generate Demand
Generate OD trip tables for various market segments/purposes using network skims and landuse data.

Outputs:

- Trip tables by segment
- **buffered_zones.csv** - A buffered version of the land use file
- **zone_production_size.csv** - A zone-indexed file with the number of trips produced per household (for each segment)
- **zone_attraction_size.csv** - The attraction size measure for each zone (for each segment)

### Assign Demand
Assign trips to road network links and calculate traffic volume.

Outputs:

- **bike_vol** - from_node, to_node, volume
