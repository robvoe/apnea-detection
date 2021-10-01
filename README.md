# Apnea detection
Apnea/hypopnea detectors for polysomnographic data, specifically for the PhysioNet 2018 dataset.

One of the two detectors works on the base of classical signal-processing and rule-based 
decisions. The other detector makes use of modern AI-based methods.

__Important:__
Before starting anything, make sure you install all necessary dependencies by preparing 
the Conda environment, see steps down below.

### Application examples & how to start
In both cases (rule-based & AI-based detector), a good start is to take a look at the 
provided notebooks under the equally named sub-folder. They demonstrate how to use the 
data processing infrastructure, how to plot nice images and how to eventually run the 
apnea/hypopnea detectors.

However, it is recommended to place the PhysioNet datasets into the sub-folder `data`. The
therein included [README.md](./data/README.md) provides more information. The aforementioned
notebooks will make use of the files stored in that folder.

### AI trainings
For those who are interested in training own AI models on the PhysioNet dataset: You 
should take a look at the files within sub-folder `ai_based`, most of all at the contained 
[README.md](./ai_based/README.md) file.

The AI training framework contained in sub-folder `ai_based` was largely built on top of
[TheFloe1995's GitHub repo](https://github.com/TheFloe1995/correct-pose) (MIT license), which 
provides a nice way to train and manage AI models as so-called _experiments_.

### Preparing Conda environment
- conda env create --file environment.yml
- conda activate apnea-detection
- conda install -y pytorch cudatoolkit=11.1 -c pytorch -c nvidia