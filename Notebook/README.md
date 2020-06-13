# PyBEC Notebook

## Installation
It is recommended that you install this within a virtualenv or conda environment. 
Python 3.5 or greater is required.

1. `pip install pybec`  This will install all of the helper functions and python
dependencies.  
2. `pip install "plotly==4.8.1" "notebook>=5.3" "ipywidgets>=7.2"`
    * these are dependencies for the notebook only and will not necessarily be installed
    with the pybec package

## Usage
1. `jupyter notebook` ==> Navigate to the pybec.ipynb and open it.
2. Make sure that the cells are run in order. Not all of them depend on others, but
it's easiest to just run them one by one. Be sure to fill out the **Point Your Files Here**
section completely.  There will be other cells that configure the plot styling.
    * Uncomment the save figure lines in the plotting cells for saving an image.  
    * The bottom plots in section 4 take a while to render.  If you shoose too high of 
    a resolution, they may also cause your notebook to crash unless you have a ton 
    of memory.  `res=40` seems to work fine, and the plotting functions do a pretty 
    good job of smoothing the heatmap with interpolation.  Feel free to drop the 
    resolution significantly when testing out different plot looks and bump up for the
    final render.
    * The **Configure the Look of the Plot** cell in Part 4 has a lot of options, 
    but it is always possible to have finer control over the plot look by adjusting
    the plotting functions themselves in the **Plot it** cell. The plotly documentation
    can be found here:
        * [Streamtube Tutorial](https://plotly.com/python/streamtube-plot/)
            * "The color of tubes is determined by their local norm, and the diameter of the field by the local divergence of the vector field."
        * [Full Streamtube Reference](https://plotly.com/python/reference/#streamtube)
        * [3D Volume Plot Tutorial](https://plotly.com/python/3d-volume-plots/) -- For Isosurfaces and Slices.
        * [Full 3D Volume Plot Reference](https://plotly.com/python/reference/#volume)
        * [3D Axes Tutorial](https://plotly.com/python/3d-axes/)
        * [Lighting Tutorial](https://plotly.com/python/v3/3d-surface-lighting/)