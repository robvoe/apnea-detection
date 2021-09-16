# General
The files and sub-folders inside this folder serve the purpose of training (and executing) AI models to 
detect respiratory events on polysomnographic data.

The here-used AI training framework mainly originates from 
[TheFloe1995's GitHub repo](https://github.com/TheFloe1995/correct-pose) (MIT license), though it
was modified/enhanced in various locations such that it fits the needs of this project. The copyright
of these modifications is held by [Robert Voelckner](https://github.com/robvoe/apnea-detection).

# Training and inference
### Preconditions
In order to be able to run training and inference, the `apnea-detection` Conda environment must be installed. If 
not yet done so, follow the instructions [here](../README.md).

### Training a model
Trainings are generally being performed by [train.py](./train.py). In order to start a training run, 
the following instructions must be followed:
1) Apply the desired train configurations to train.py by modifying the script
2) Navigate to the root of the apnea-detection project: `cd /path/to/apnea-detection`
3) Activate apnea-detection Conda environment: `conda activate apnea-detection`
4) Run `python3 -m ai_based.train`

Training runs are organized as experiments; experiment results (trained models, info on the train run, 
etc.) can be found in sub-folder `<project-root>/ai_based/results`.

### Inspection of trained models & model inference
The notebooks in folder `<project-root>/notebooks` serve the purposes of 
- analyzing trained models,
- giving hints on how to facilitate trained models to detect respiratory events.