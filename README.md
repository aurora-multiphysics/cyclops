# Cyclops

Cyclops is a sensor suite optimisation program, designed for fusion engineering experiments.

Cyclops takes a simulated experiment and searches for the optimal layout of sensors whose readings allow for an accurate reconstruction of the full field information, while remaining robust to sensor failures.

# Installation

## Linux

Create a virtual environment and activate it. (Skip this step if you intend to install cyclops into an existing python environment.)
```
python -m venv venv
source venv/bin/activate
```

### Standard install
To install `cyclops`, run the following:
```
# Ensure `build` package is installed.
pip install build --upgrade

# Build the distribution package.
python -m build

# Install the built distribution.
pip install ./dist/*.whl
```

Cyclops can now be imported into python as:
```
import cyclops
```

To test the installation, run `python -c "import cyclops"`. If the installation was unsuccessful, an error will be raised.

### Editable install (developer mode)
Developers may wish to create an editable installation. This allows changes to the source code to immediately take effect without the need to re-package and re-install cyclops. This can be useful when running tests and other scripts.
To install `cyclops` this way, run the following:
```
# Install as an editable installation.
pip install --editable .
```

### Latex plots

To enable Latex style plots, install latex via:
```
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```
Then initialise the plot manager as `PlotManager(latex_mode=True)`.

## Windows

> Windows installation is untested.

## MacOS

> MacOS installation is untested.


# Getting started

To start using cyclops, please refer to the included tutorials in `cyclops/tutorials/`.

Example scripts are also included in `cyclops/examples/`.

# Contributors

Dom Harrington, UK Atomic Energy Authority

Luke Humphrey, UK Atomic Energy Authority

Lloyd Fletcher, UK Atomic Energy Authority

# Citing Cyclops

If using Cyclops in your research, please be sure to cite its key dependencies. Information on what to cite can be found on each package's website as follows:

- Scikit-learn (https://scikit-learn.org/stable/about.html#citing-scikit-learn)
- Meshio (https://github.com/nschloe/meshio/blob/main/CITATION.cff)
- NumPy (https://numpy.org/citing-numpy/)
- SciPy (https://scipy.org/citing-scipy/)
- Matplotlib (https://matplotlib.org/stable/users/project/citing.html)
- SciencePlots (https://github.com/garrettj403/SciencePlots#citing-scienceplots)




