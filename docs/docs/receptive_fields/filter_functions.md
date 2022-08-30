# Overview

## Coordinates

* Coordinates are the spatial and temporal values that form the "*canvas*" from which spatial and temporal filter functions are calculated.  
	- That is, the coordinates are arrays of values that provide the basic inputs to the spatial and temporal functions used to create filters and stimuli.
* More practically, they are `arrays` of uniform/regular values, defined by a `resolution`, _which is the difference between adjacent values_, and an `extent`, _which is maximum and/or minimum value_.

### Spatial Coordinates

For simplicity and predictability, the following constraints have been enforced in the generation of spatial coordinates:

* **Zero-Centred**: The central pixel is always zero in both `x` and `y` coordinates/axes.
	- _Corollary_: There is a central pixel, which means that the number of coordinates are always odd, with a central pixel "flanked" by two symmetrical (see below) subsets each with an even number of coordinates.
* **Symmetrical**: Around the central zero pixel values extend equally and symmetrically into the positive and negative number range and extend to the same absolute value, but positive in one direction and negative in the other.
* **Snapped-Extent**: The extent (or min/max) of the coordinates are "_snapped_" to be a whole integer multiple of the resolution of the coordinates.
	- This is intended to make slicing portions of the stimulus easier.

#### Process and Functions

* [`mk_spat_coords_1d`][receptive_field.filters.filter_functions.mk_spat_coords_1d] is the primary *underlying* or *behind the scenes* function.  It generates a one dimensional array of coordinates that meet the constraints listed above using the provided resolution and extent arguments.
	- The extent is _snapped_ by using `mk_rounded_spat_radius`, which treats the `spat_ext` argument as a "*width*" or "*diameter*" and returns a "*radius*" (by dividing the `spat_ext` value by `2`) that is *snapped* to the nearest whole integer of the `spat_res` argument.  This *snapped* value is in the same unit as `spat_res`, is always *snapped* upward of the original value, and is an `integer`.
	- With the appropriately *snapped* extent (now as a *radius*), `np.arange` is used to create the array of values, by starting at the negative of the *radius* and going to the positive of the sum of the radius and the resolution (where `np.arange` does not include the `stop` value, so stopping at the sum of the *radius* and resolution *ensures the symmetry as defined above*).  The step is of course the `spat_res` value.
	- As integers are used in the `np.arange` and the start and stop values are whole integer multiples of step, the resulting array will also comprise integers.
* `mk_sd_limited_spat_coords` is a small wrapper around `mk_spat_coords_1d` for the purposes of allowing the user to forego the need for a manually generated extent value and instead use the definition of a relevant spatial filter to limit the extent of the coordinates in such a way as to ensure that the whole of the filter "fits" into the coordinates canvas (with only negligible loss of the extreme tails, of course, such filters being mostly asymptotic like the `gaussian`).  This function is mostly likely to be used by other functions in this module as a way to generate one dimensional spatial coordinates.
	- This interface allows the user to alternatively provide an extent argument like with `mk_spat_coords_1d` or instead provide an `sd` argument.  IE, a `standard deviation` of a putative spatial filter.  The extent is then calculated from the `sd` value, multiplying it out by the `sd_limit` argument which takes a default value taken from the `settings` (most likely `5` as this is a good point at which to cut off the filter).  And then doubling it, as extent arguments define the *width* of the spatial coordinates, while a `standard deviation` of a spatial filter is a radial metric.
* `mk_spat_coords` is the primary function for generating the *coordinates canvas*, using `np.meshgrid` to create two dimensional arrays that contain the `x` and `y` values for the full grid of coordinates.  It takes the same arguments as `mk_sd_limited_spat_coords` and uses the same under the hood.
	- The `x` and `y` one dimensional coordinates are created separately and then fed to `np.meshgrid`.
	- The `y` coordinates are the same as the `x` but _reversed_.  This is so that the "*top*-*left*" corner of the canvas will represent the *intuitive* "*top*" of an image or graph.  That is, by being reversed, the highest value of the `y` 1D array will be the first value (at index `0`).  This is the opposite for the `x`.  Once the 2D coordinates are formed with `meshgrid`, this means that the _first_ row of values will have the highest `y` values, as though they are at the *top* of a `y`-`x` graph.  The first column will have the lowest `x` values, as they would represent the far left of the graph.
	- _In other words:_ the `x` coordinates go up left-to-right, the `y` coordinates go up bottom-to-top; as occurs in mathematical 2D graphs.
	- The way `np.meshgrid` orders the dimensions is inline with the typical ordering of the axes in `numpy`, where, for a 2D array for instance, the rows are the first axis and the columns the second (`C order` or `Row major`).  In terms of `y` and `x`, this means that the `y` coordinates vary along the first axis of the array and `x` the second.  EG: `X[[0, 1, 2], 0]` will three numbers with the same value, but `X[0, [0, 1, 2]]` will show multiple and different x coordinates. Vice versa for `Y[[0, 1, 2], 0]` (coordinates vary) and `Y[0, [0, 1, 2]]` (do not vary).

#### Dependency Chart

```
"-->" ~ "Uses or depends on"
"(A, B)" ~ "Provides A and B as arguments to"
 													  (res, spat_ext/spat_filt args)
* mk_spat_coords <--- (res, sd)                       * spat_filt_size_in_res_units --> (n units)
| (res, sd)											  |
V 													  | 
* mk_sd_limited_spat_coords 						  |
|      \  											  |
|       \ (sd) 										  |
| (res)  * mk_spat_ext_from_sd_limit (rounds up) * <--|
|       / (ext) 									  |
V      / 											  |
* mk_spat_coords_1d 								  |
|      \ 											  |
|       \ (res, ext) 								  |
| (res)  * mk_rounded_spat_radius * <-----------------|
|        |           \ (res, ext / 2)
|       / <---------- * round_coord_to_res (rounds up)
V      / (radius)
* np.arange 
```



### Temporal Coordinates

* Generally simpler than spatial as always one dimensional.
* `mk_temp_coords` is the primary function and is analogous to `mk_sd_limited_spat_coords`.  
	- The extent of the coordinates always run from `0` to the provided extent value.
	- The extent of coordinates can also, as with `mk_sd_limited_spat_coords`, be defined by a `tau` argument which represents the breadth of a putative temporal filter (the time constant), and, the number of multiples of this time constant that will be used the limit for the temporal coordinates, which again defaults to a value defined in the settings (most likely `10`).
	- The resulting coordinates will range from `0` (inclusive) to a value that is at least as great as the provided extent using, again, `np.arange()`


### Spatio-Temporal Coordinates

* Like the full grid of values of `spatial` coordinates produced by `mk_spat_coords`, `mk_spat_temp_coords` does the same but also includes a temporal dimensions.
* As a result, the arrays are 3 Dimensional.
* The function takes the super-set of arguments required by `mk_spat_coords` and `mk_temp_coords`, including the putative filter metrics `sd` and `tau`.
* The three 1 Dimensional vector arrays provided to `meshgrid` are produced by `mk_sd_limited_spat_coords`, in the case of the `x` and `y` vectors, and by `mk_temp_coords` for the temporal.  As with `mk_spat_coords`, the `y` vector is a reversed copy of `x`.




::: receptive_field.filters.filter_functions
