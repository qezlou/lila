# Line Intensity map X Ly-Alpha forest  (LILA)
Forecasting the cross-orrelation between Line Intensity map and Ly-Alpha Forest surveys

This repo containts a python package written for Qezlou et al. 2023 which forecasts the 3D auto and cross power-spectra for Ly-alpha forest, CO Line intensity map [COMAP](https://comap.caltech.edu/) and galaxy redshift surveys. 

#### The story behind the name:
Lila is a name for girls with Arabic, Persian, and Hindi roots, meaning "night" or "play". One famous reference to this name is the tale of ["Layla" (another spelling) and "Majnun"](https://en.wikipedia.org/wiki/Layla_and_Majnun), an ancient Arab story about the poet Qays from the 7th century and his beloved Layla. The story was later passed on to Persian culture through a beautiful poem written by Nizami Ganjavi between 584-1188. "Layla and Majnun" is also called the Eastern version of Romeo and Juliet.

## Installation
It requires python version < 3.9

You can install this simply by:

1. pip :  `pip install lali`
2. clonning this repo and then installing : 
```
git clone https://github.com/qezlou/lali.git
cd lali
python -m pip install -e .
```

## Contetnts:
#### Modules:
- `compa.py`: A module to make mock observations for [COMAP-Y5](https://comap.caltech.edu/)
- `lim.py`: The base module for making LIM mocks, e.g. `comap.py` inherits from this. 
- `mock_lya.py`: A module to make mock observations for the 3D Ly-alpha forest (tomography)
- `mock_galaxy`: A module to make mock observations for galaxy redshfit surveys 
- `stats.py`: Takes the mock observations as input and calculates the 3D auto and cross power-spectra
- `inference.py`: Takes the forecast power spectra and runs inference on the paramters for the biased linear power spectra. 
- `plot.py`: A few plotting tools.

#### helper scripts:
- `get_gal.py`: Get the mock 3D power spectra for auto CO, galaxy and CO X galaxy.
- `get_lya.py`: Get the mock 3D power spectra for auto CO, Lya forest and CO X Lya forest.
- `get_latis_source_pk.py`: Get the projected 2D power spectrum of the sources in [LATIS](https://ui.adsabs.harvard.edu/abs/2020ApJ...891..147N/abstract). **Note:** data used here for LATIS are not publicly available yet, so you can skip this code for now. 

#### Notebooks:
- `galaxy_selection.ipynb`: Analyzing the [HSC photometric obsevration](https://www.clauds.net/available-data) to obtain:
1.  The median redshift uncertainties 
2.  The mass completness with halo abundance matching technique. 
- `SN_results.ipynb`: The results for the forecast S/N ratio of the CO, CO X Lya and CO X Galaxies.
- `Inference.ipynb`: The results for the inferencee on the biased linear power spectrum parameters. 

## Data: 

All the simulated data are available here on Zenodo:

