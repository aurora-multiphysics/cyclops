# cyclops

Cyclops is a sensor placement optimisation program. Originally designed for monoblocks at the bottom of nuclear fusion reactors it searches for the optimal layout of thermocouples (with IR cameras and DIC sensors to come) to maximise both the accuracy and reliability of the temperature field reconstruction.

After reading this, go straight to manual/overview.ipynb to find out how to use the software.

Requirements: Python 3.x, Latex (Latex is very strongly recommended but not 100% essential - see installation step 2 to get around it)


# Installation

Please follow the two steps below very carefully. 

1. Install the requirements as below

python3 -m venv venv
source venv/bin/activate
pip install -r requirements/app.txt

2. Install Latex on your device to let latex style graphs be plotted. If you don't care about the latex style graphs or can't install latex then go to src/graph_management.py and then backspace the line in GraphManager.__init__() about plt.style.use('science'). Also don't bother importing scienceplots.

You are now ready to go!


# Citations needed:

1. Matplotlib
2. Meshio
3. NumPy
4. SciencePlots (for Latex style graphs)
5. Scikit-learn
6. SciPy


# Contributers

Dom Harrington

Luke Humphrey


