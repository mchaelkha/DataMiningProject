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
