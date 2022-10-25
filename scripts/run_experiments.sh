#!/usr/bin/env bash
# This script runs the experiments for the paper "Deep Learning for Rapid Landslide Detection using Synthetic Aperture Radar (SAR) Datacubes"
# Run the script from the base directory as follows: bash scripts/run_experiments.sh

ds_path="..../hokkaido_japan.zarr" # Path to the dataset downloaded from zenodo (see README.md)

 for timestep_length in {1..4..1}
 do
   echo "Experiment with timestep_length ${timestep_length}"
   # only vv
   python src/main.py --input_vars "vv_before,vv_after,vh_before,vh_after" --timestep_length ${timestep_length} --ds_path ${ds_path}
   # only vh
   python src/main.py --input_vars "vv_before,vv_after,vh_before,vh_after" --timestep_length ${timestep_length} --ds_path ${ds_path}
   # vv + vh
   python src/main.py --input_vars "vv_before,vv_after,vh_before,vh_after" --timestep_length ${timestep_length} --ds_path ${ds_path}
   # SAR + DEM
   python src/main.py --input_vars "vv_before,vv_after,vh_before,vh_after,dem,dem_curvature,dem_slope_radians,dem_aspect" --timestep_length ${timestep_length} --ds_path ${ds_path}

done

# only DEM
python src/main.py --input_vars "dem,dem_curvature,dem_slope_radians,dem_aspect" --timestep_length ${timestep_length} --ds_path ${ds_path}
