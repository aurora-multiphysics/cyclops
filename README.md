# Cyclops

Cyclops is a sensor suite optimisation program, designed for fusion engineering experiments.

Cyclops takes a simulated experiment and searches for the optimal layout of sensors whose readings allow for an accurate reconstruction of the full field information, while remaining robust to sensor failures.

After reading this, go to '*tutorials/0 overview.ipynb*' to find tutorials on how to use the software.


# Installation

Clone the repository. Then add in a `results` folder at the `cyclops/results` level. This folder is where your results will be stored.

## Linux

1. Install the requirements as below. You will need Python 3.8 or higher.

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

2. Install Latex on your device to let latex style graphs be plotted. 

`sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super`

If you don't care about the latex style graphs or can't install latex then go to src/graph_management.py and then change the line in `GraphManager.__init__()` about `plt.style.use('science')` to the line below.

`plt.style.use(['science','no-latex'])`


## Windows

1. Install the requirements as below. You will need Python 3.8 or higher.

`python3 -m venv venv`

`venv/scripts/activate`

`pip install -r requirements.txt`

2. Install Latex on your device to let latex style graphs be plotted. 

On Windows I recommend MikTex https://miktex.org/.

If you don't care about the latex style graphs or can't install latex then go to src/graph_management.py and then change the line in `GraphManager.__init__()` about `plt.style.use('science')` to the line below.

`plt.style.use(['science','no-latex'])`


## Mac

Note that the mac installation is untested.

1. Install the requirements as below. You will need Python 3.8 or higher.

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

2. Install Latex on your device to let latex style graphs be plotted. 

On MacOS I recommend MacTex https://www.tug.org/mactex/.

If you don't care about the latex style graphs or can't install latex then go to src/graph_management.py and then change the line in `GraphManager.__init__()` about `plt.style.use('science')` to the line below.

`plt.style.use(['science','no-latex'])`


# Citations needed:

1. Matplotlib (https://matplotlib.org/stable/users/project/citing.html)
2. Meshio (https://github.com/nschloe/meshio/blob/main/CITATION.cff)
3. NumPy (https://numpy.org/citing-numpy/)
4. SciencePlots (https://github.com/garrettj403/SciencePlots)
5. Scikit-learn (https://scikit-learn.org/stable/about.html#citing-scikit-learn)
6. SciPy (https://scipy.org/citing-scipy/)


# Contributers

Dom Harrington, UK Atomic Energy Authority

Luke Humphrey, UK Atomic Energy Authority

Lloyd Fletcher, UK Atomic Energy Authority


