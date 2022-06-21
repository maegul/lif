# Overview

## Coordinates

* Coordinates are the spatial and temporal values that form the "*canvas*" from which spatial and temporal filter functions are calculated.  That is, the coordinates are arrays of values that provide the basic inputs to the spatial and temporal functions used to create filters and stimuli.
* More practically, they are `arrays` of uniform/regular values, defined by a `resolution`, which is the difference between adjacent values, and an `extent` which is maximum and/or minimum value.

### Spatial Coordinates

For simplicity and predictability, the following constraints have been enforced in the generation of spatial coordinates:

* **Zero-Centred**: The central pixel is always zero in both `x` and `y` coordinates.
* **Snapped-Extent**: The extent (or min/max) of the coordinates are "_snapped_" to be an integer multiple of the resolution of the coordinates.
	- This is intended to make slicing portions


::: receptive_field.filters.filter_functions
