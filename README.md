## Point Pattern Analysis in PySAL

Statistical analysis of planar point patterns

**REORG IN PROGRESS**

This is work in progress towards [PEP 13](https://github.com/pysal/pysal/wiki/PEP-13:-Refactor-PySAL-Using-Submodules)


### History

#### 2016-12-23

	mkdir pysal_reorg
	cd pysal_reorg
	git clone git@github.com:sjsrey/pysal.git
	cp -R pysal pysal-copy
	cp -R pysal-copy pysal-points
	cd pysal-points
	git fetch origin
	git branch -a
	git checkout -b points origin/points_contrib
	git filter-branch --subdirectory-filter pysal/contrib/points -- -- all
	git remote -v
	git remote add upstream git@github.com:cgiscourses/pysal-points.git
	git push -u upstream points
	git branch -d master
	git checkout -b master
	git push upstream master


After these steps, on the Github settings for the repos change the default branch from `points` to `master`.



### Introduction

This PySAL  module is intended to support the statistical analysis of planar point patterns.

It currently works on cartesian coordinates. Users with data in geographic coordinates need to project their data prior to using this module.


### Examples

* [Basic point pattern structure](notebooks/pointpattern.ipynb)
* [Centrography and Visualization](notebooks/centrography.ipynb)
* [Marks](notebooks/marks.ipynb)
* [Simulation of point processes](notebooks/process.ipynb)
* [Distance based statistics](notebooks/distance_statistics.ipynb)

### Installation

#### Requirements

- PySAL 1.11+
- Pandas 0.17.0+
- shapely
- descartes

### TODO

- Enhance internal data structures
- Remove pysal and replace dependency with pysal_core


[contrib]: http://pysal.readthedocs.org/en/latest/library/contrib/index.html
[GeoPandas]: http://geopandas.org
[pandas]: http://pandas.pydata.org

