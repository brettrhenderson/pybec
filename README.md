PyBEC
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/brettrhenderson/pybec.svg?branch=master)](https://travis-ci.com/brettrhenderson/pybec)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/PyBEC/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/PyBEC/branch/master)


Python package for extracting and manipulating Born Effective Charges from QuantumEspresso Output


For Localhost
=============

Installation
------------

1) Install python 3.5 + (I've tested with 3.7, but I think it should work with some earlier versions)
2) Make a virtual environment for your project
    1) create a folder to store virtual environment: mkdir ~/virtualenvs
    2) create the virtualenv: virtualenv -p python3.7 ~/virtualenvs/born_charges
    3) activate the virtualenv: source ~/virtualenvs/born_charges/bin/activate
        - should now see a (born_charges) in front of bash prompt
    4) upgrade pip: pip install --upgrade pip
3) Install required modules
    1) pip install -r requirements.txt
    2) pip install pykrige

4) deactivate your virtualenv with command 'deactivate'

5) Place the files in a good location. This can be anywhere in your home directory and you could add the scripts to your PATH to be able to access them elsewhere.  Just note that the Jupyter notebook and the plot_becs.py script both import bec_helpers, so you should make sure that bec_helpers is in the same folder as them.


Operation: Jupyter Notebook
---------------------------

1) Activate the virtual environment created above:
    source ~/virtualenvs/born_charges/bin/activate

2) open jupyter notebook: jupyter notebook

3) run through the sections and change parameters to point to your files


Operation: Command Line <a name="clilocal"></a>
-----------------------

1) Activate the virtual environment created above:
    source ~/virtualenvs/born_charges/bin/activate

2) Run the program

    ```python plot_becs.py becvdist --noefield 7_zero_field.out --clampedion 8_efield_electron_pol.out --xyz Ag8_MgO_333.xyz --interactive```

    * 'becvdist' : the name of the operation to perform.  Can be:
        1) becvdist: plot the Born effective charges against the distance from nanoparticle centroid
        2) plotheatmap: interpolate the Born effective charges and plot them as a heatmap.
           Used to verify that the interpolation worked
        3) exportheatmap: add the heatmap as volumetric data to an existing xsf file, or copy that
           xsf file and the volumetric data to a new  xsf file.
    * the other three arguments point toward the output files you will need. If using --interactive,
      you can omit them and will be prompted for them, but I have not yet implemented tab completion
      in the prompts, so it might be easiest to just put them in here.
    * interactive means that you will be prompted for any other information that is needed.
    * all of these other bits of info can be passed directly at the command line, but interactive
      makes sure you don't forget anything.

3) run `python plot_becs.py -h` for a full list of command-line options, or just run
   `python plot_becs.py [becvdist, plotheatmap, exportheatmap] --interactive` for the easiest option


For Cedar
=========

Installation
------------

1) Log in to cedar
2) load the correct version of python:
    module load python/3.7.0
        - can verify by then running python --version

3) create a virtual environment
    a. create a folder to store virtual environment: mkdir -p ~/virtualenvs/born_charges
    b. create the virtualenv: virtualenv --no-download ~/.virtualenvs/born_charges
    c. activate the virtualenv: source ~/.virtualenvs/born_charges/bin/activate
        - should now see a (born_charges) in front of bash prompt
    d. upgrade pip: pip install --upgrade pip

4) install required modules
    a. install the requirements list: pip install --no-index -r requirements_cedar.txt
    b. install pykrige: pip install pykrige
    c. install dask toolz: pip install dask[array] --upgrade


Operation: Generate the Data
----------------------------

I haven't looked at any way to run Jupyter Notebooks on cedar, so for now command line is the only way.

**BEC v. Distance from Centroid**
_________________________________

1) module load python/3.7.0
2) activate your virtual environment
    source ~/virtualenvs/born_charges/bin/activate

3) To plot the born effective charges against the distance from the nanoparticle, just follow the procedure outlined above
under [2) Run the Program](#clilocal) on your login node.

**Plot Heatmap and Export Heatmap**
___________________________________

For these operations, some of the interpolation methods can use up over 1 GB of RAM if your resolution is over 50 points per side.  This makes it best to run them as jobs on compute nodes.  The script plot_bec.sh, is a job submission script that can be submitted with sbatch.  Edit this file with the appropriate variables and then submit.  Any plots will be saved in the folder you specify, to be displayed later.

Display Plots
-------------

Plots can be displayed with the script disply.py if you log in to cedar with the -Y flag. You do, however, need to load both python and PyQt, which is a python wrapper for the Qt GUI library.

1) Load python: module load python/2.7.14
2) Load Qt: module load qt/5.10.1  (don't load the newest qt5 as it seems to not work)
3) python display.py plot.png
    - plot.png is the name of the image file to display.  It can be either a PNG or SVG file.




### Copyright

Copyright (c) 2020, Brett Henderson


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
