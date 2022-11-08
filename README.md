# Deep Learning for Rapid Landslide Detection using Synthetic Aperture Radar (SAR) Datacubes

Repository for the paper [Deep Learning for Rapid Landslide Detection using Synthetic Aperture Radar (SAR) Datacubes](https://arxiv.org/abs/2211.02869)

## Installing the requirements
To run the experiments presented in the paper make sure to install the requirements.

`pip install -r requirements.txt`

## Downloading the data 

Download the data from [Zenodo](https://doi.org/10.5281/zenodo.7248056). Particularly, the [hokkaido datacube](https://zenodo.org/record/7248056/files/hokkaido_japan.zip) is needed.

## Running the experiments

To reproduce the experiments from the paper run the script  

`bash scripts/run_experiments.sh`

**IMPORTANT:** After, decompressing the downloaded hokkaido cube, make sure to add datacube path to the script before running it.

## Notes

The experiments have run on an NVIDIA V100 GPU in Google Cloud.

## Citation

If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2211.02869):

```
@misc{https://doi.org/10.48550/arxiv.2211.02869,
  doi = {10.48550/ARXIV.2211.02869},
  
  url = {https://arxiv.org/abs/2211.02869},
  
  author = {Boehm, Vanessa and Leong, Wei Ji and Mahesh, Ragini Bal and Prapas, Ioannis and Nemni, Edoardo and Kalaitzis, Freddie and Ganju, Siddha and Ramos-Pollan, Raul},
  
  keywords = {Signal Processing (eess.SP), Computer Vision and Pattern Recognition (cs.CV), Image and Video Processing (eess.IV), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Deep Learning for Rapid Landslide Detection using Synthetic Aperture Radar (SAR) Datacubes},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```


## Acknowledgements

This work has been enabled by the Frontier Development Lab Program (FDL). FDL is a collaboration between SETI Institute and Trillium Technologies Inc., in partnership with the Department of Energy (DOE), National Aeronautics and Space Administration (NASA), and the U.S. Geological Survey (USGS). The material is based upon work supported by NASA under award No(s) NNX14AT27A.
