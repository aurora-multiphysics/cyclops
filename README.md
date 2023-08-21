# cyclops

Cyclops is a sensor placement optimisation program. Originally designed for monoblocks at the bottom of nuclear fusion reactors it searches for the optimal layout of thermocouples (with IR cameras and DIC sensors to come) to maximise both the accuracy and reliability of the temperature field reconstruction.

If you plan on using it, we recommend you read the manual beforehand (in the manual folder).

Requirements: Python 3.x, Latex (Latex is optional - see installation step 2 to get around it)


# Installation

Please follow the two steps below very carefully. 

1. Install the requirements as below

python3 -m venv venv
source venv/bin/activate
pip install -r requirements/app.txt

2. Install Latex on your device to let latex style graphs be plotted. If you don't care about the latex style graphs and/or can't install latex then go to src/graph_management.py and then backspace the line in GraphManager.__init__() about plt.style.use('science').

You are now ready to go!


# Contributers

Dom Harrington

Luke Humphrey


