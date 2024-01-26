# Seating Posture Detection



## Contents

- [Installation](#installation)
- [Use](#use)


## Installation

To use correctly the different parts of the code, you should install the required modules with the command: 
$ pip install -r requirements.txt

## Use

To get the different clusters, run the notebook the files Kmean.py, atoms.py and dadil.py under md_clustering/Experiments. Make sure to check and change the directories you want to save the results in in each file.
You will need a gpu for computing with posture data. We recommend batching your data before hand.

To clean the outliers, please see the notebooks in Data under Posture_Data.

For the interface, please connect the mat and proceed to run the script in the interface directory.