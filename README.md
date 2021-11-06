# Toponym Georeferencing Paper - Replication Code

This repository contains the algorithm and simulation code for the paper "Toponym-assisted map georeferencing: Evaluating the use of toponyms for the digitization of map collections", by Karim Bahgat and Dan Runfola (2021). 

For the latest version including possible fixes to these replication scripts, go to: https://github.com/karimbahgat/ToponymGeoreferencingPaper. 

## Installation

The code and packages in this repository works on Python 2.7 and 3.x. 

The main package for map georeferencing is contained in the "automap" package folder. Note that this package is intended solely for replicating the implementation and results used in the article -- a more user-friendly standalone georeferencing package may be announced at a later point. 

To install this package and all the necessary dependencies on your machine, write the following from the commandline:

```
pip install git+https://github.com/karimbahgat/ToponymGeoreferencingPaper
``` 

### Tesseract

The text recognition part requires that you have Tesseract installed on your machine. For information on how to install Tesseract, see https://tesseract-ocr.github.io/tessdoc/Installation.html. 

### Data dependencies

The replication scripts in this repository are dependent on several external data files. Some of these data dependencies are not included in this repository due to file size restrictions, but instead have to be downloaded as a zipfile from URL-TO-BE-ADDED-LATER. The contents of the zipfile should be extracted to the "data" folder. 

## Replicating the results

With the necessary packages, scripts, and data installed, it's time to run the replication scripts. 

### Simulated maps

The "simulations" folder of this repository contains the scripts necessary to replicate the results for the automated map georeferencing of simulated maps presented in the article. 

Be forewarned that the simulations generates very large volumes of map image data and will take days or weeks to process depending on the number of CPU Cores (the number of CPU cores can be specified in the "PARAMETERS" section near the top of each script). Specifically, the scripts should be run in the following order:

- `1) generate maps.py` (generates the simulated maps and outputs them into `simulations/maps`)
- `2) georeference.py` (uses the automap package to automatically georeference the simulated maps and outputs the results in `simulations/outputs`)
- `3) error assessment.py` (generates error metrics and outputs them as files ending in `_error.json` in `simulations/outputs`)
- `4) error analyze.py` (collects all the information about simulated maps, map parameters, georeferencing results and errors, and outputs them as json strings in an sqlite database in `simulations/analyze/stats.db`. Also generates some but not all of the article figures in `simulations/analyze/figures`).

### Real-world maps

The "realworld" folder of this repository contains the scripts necessary to replicate the results for the automated map georeferencing of real-world maps presented in the article. 

To replicate the results of the real-world maps, the scripts should be run in the following order: 

