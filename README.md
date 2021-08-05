# apnoe-detection
Apnoe detector for polysomnographic data

# Sources
- Base of the AI project structure: https://github.com/TheFloe1995/correct-pose (MIT license)
- Data & basic data import code: https://physionet.org/content/challenge-2018/1.0.0/

# Preparing Conda environment
- conda env create --file environment.yml
- conda activate apnoe-detection
- conda install pytorch cudatoolkit=11.1 -c pytorch -c nvidia