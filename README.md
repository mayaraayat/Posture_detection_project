# Seating Posture Detection



## Contents

- [Installation](#installation)
- [Use](#use)
- [Acknowledgements](#acknowledgments)


## Installation

To use correctly the different parts of the code, you should install the required modules with the command: 
$ pip install -r requirements.txt

You will need two environments: one with pytorch-lightning 1.9 to compute atoms.py for initializing the atoms, and pytorch-lightning 2.0 or more for the dictionary learning step in dadil.py

## Use

To get the different clusters, run the notebook the files Kmean.py, atoms.py and dadil.py under md_clustering/Experiments. Make sure to check and change the directories you want to save the results in in each file.
You will need a gpu for computing with posture data. We recommend batching your data before hand.

To clean the outliers, please see the notebooks in Data under Posture_Data.

For the interface, please connect the mat and proceed to run the script in the interface directory.

## Acknowledgements

We are grateful for Dr.NGOLE MBOULA Fred Maurice for his support and his help throughout the project. 
We thank our teachers Wassila Ouerdane and Jean Philippe Poli who provided technical support and were there to accompany us at each step of the project.
Special thanks to Anas Hattay for his help and advice to better manage the project.
