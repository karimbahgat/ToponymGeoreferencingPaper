"""
# ClassyPie

Python toolkit for easily classifying (grouping) data values.

## Introduction

Data classification algorithms are commonly used to group together data
values in order to simplify and highlight different aspects of the data
distribution.

ClassyPie implements and gives users easy access to many of the most
commonly used data classification algorithms. 
Can be used on any sequence of values, including objects by
specifying a key-function to obtain the value.

The library was originally built as a convenient high-level wrapper around
Carston Farmer's "class_intervals.py" script for QGIS, but has since been
improved and expanded with several new algorithms and new convenience functions.

## License

This code is free to share, use, reuse, and modify according to the MIT
license, see license.txt

## Credits

- Karim Bahgat
- Carston Farmer

"""


__version__ = "0.1.0"


from .main import *


