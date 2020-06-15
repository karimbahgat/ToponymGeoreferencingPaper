# AutoMap

This repository contains the algorithm and simulation code for the paper "A Toponym-Based Approach to the Automated Georeferencing of Physically Mapped Documents: The TAAG Algorithm", by Karim Bahgat and Dan Runfola (2020). 

For any subsequent updates or bug fixes, go to the GitHub repository at https://github.com/karimbahgat/AutoMap. 

## Installation

The code and packages in this repository only works on Python 2.7. 

The main package for map georeferencing is contained in the "automap" package folder. Note that this version of the package is intended solely for replicating the results of the article -- a more user-friendly standalone georeferencing package will be announced at a later point, see https://github.com/karimbahgat/AutoMap for updates. 

To install the "automap" package on your machine, write the following from the commandline:

```
pip install git+https://github.com/karimbahgat/AutoMap
``` 

### Tesseract

The text recognition part requires that you have Tesseract installed on your machine. This can be done via Anaconda, by typing:

```
conda install -y -c conda-forge tesseract
```

## Simulation replication

Once the "automap" package is installed, the "simulations" folder of this repository contains the scripts necessary to replicate the results for the automated map georeferencing parts of the article. Start by downloading and extracting the "simulations" folder to your local computer. 

### Data dependencies

The simulation scripts are dependent on several external data files. These data dependencies are not included in this repository due to file size restrictions, but instead have to be downloaded as a zipfile from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3894607.svg)](https://doi.org/10.5281/zenodo.3894607). The contents of the zipfile should be extracted to the "simulations/data" folder. 

### Replicating the results

With the necessary packages, scripts, and data installed, it's time to run the simulations. Be forewarned that the simulations generates very large volumes of map image data and will take days or weeks to process depending on the number of CPU Cores (the number of CPU cores can be specified in the "PARAMETERS" section near the top of each script). Specifically, the scripts should be run in the following order:

- `1) generate maps.py` (generates the simulated maps and outputs them into "`simulations/maps`)
- `2) georeference.py` (uses the automap package to automatically georeference the simulated maps and outputs the results in `simulations/outputs`)
- `3) error assessment.py` (generates error metrics and outputs them as files ending in `_error.json` in `simulations/outputs`)
- `4) error analyze.py` (collects all the information about simulated maps, map parameters, georeferencing results and errors, and outputs them as json strings in an sqlite database in `simulations/analyze/stats.db`. Also generates some but not all of the article figures in `simulations/analyze/figures`).



