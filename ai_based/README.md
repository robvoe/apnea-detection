# General
The files and sub-folders inside this folder serve the purpose of training (and executing) AI models to 
detect respiratory events on polysomnographic data.

Parts of the here-used AI training structure originates from 
[TheFloe1995's GitHub repo](https://github.com/TheFloe1995/correct-pose), which entirely stands under the 
MIT license. 

# Training and inference
### Preconditions
In order to be able to run training and inference, the `apnea-detection` Conda environment must be installed. If 
not yet done so, see instructions [here](../README.md).

### Training a model
Trainings are generally being performed using the file [train.py](./train.py). In order to start a training run, 
the following instructions must be followed:
1) Navigate to the root of the apnea-detection project: `cd /path/to/apnea-detection`
2) Activate apnea-detection Conda environment: `conda activate apnea-detection`
3) Run `python3 -m ai_based.train`

Training runs are organized as experiments; trained models can be found in sub-folder `./ai_based/results`.

### Running inference
The notebooks in (project root) folder `/path/to/apnea-detection/notebooks` show how to facilitate trained models 
to detect respiratory events.