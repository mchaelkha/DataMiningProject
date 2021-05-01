# DataMiningProject
Dataset can be found here: https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
## Requirements
- Python 3
  - .7 .8
## Installation

### Windows
- `pip install -r requirements.txt`
- Download pyproj from https://www.lfd.uci.edu/~gohlke/pythonlibs/#basemap.
  Make sure to install the corresponding Python version.
  Then run `pip install PYPROJ_FILE.whl`
- Download basemap from https://www.lfd.uci.edu/~gohlke/pythonlibs/#basemap.
  Make sure to install the corresponding Python version.
  Then run `pip install BASEMAP_FILE.whl`
- Basemap has been tested with version 3.7, 3.8 of Python

### Mac
- `pip install -r requirements.txt`
- Download latest basemap release from https://github.com/matplotlib/basemap/releases/.
  Unzip the X.Y.Z source tar.gz file and cd to the basemap-X.Y.Z directory.
  Then run `python setup.py install`.

## Usage

### Data Cleaning
- Some visualizations require external cleaning to do this run `python data_cleaner.py`
- Before running ensure you have the downloaded the Dataset described above and named it `Motor_Vehicle_Collisions_-_Crashes.csv`.
- A file called `Clean_Motor_Vehicle_Collisions_-_Crashes.csv` should be generated after running the data cleaning script.

### Generating Visualizations
- Visualizations can be generated via running `python analytics.py`
- Before running ensure you have the downloaded the Dataset described above and named it `Motor_Vehicle_Collisions_-_Crashes.csv`.
- Before running ensure you have performed the steps described under Data Cleaning

