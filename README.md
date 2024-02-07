<div align="center">

# SenNet + HOA - Hacking the Human Vasculature in 3D solution
https://www.kaggle.com/competitions/blood-vessel-segmentation/overview


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

Codebase for SenNet + HOA - Hacking the Human Vasculature in 3D solution (#d Place)

## Download data 
```bash
kaggle competitions download -c blood-vessel-segmentation
```

## Download additional data from the 
```
http://human-organ-atlas.esrf.eu
```

If you want to skip the initial training without pseudo, you can download the images with pseudo-labels from Kaggle.
```
kaggle datasets download -d igorkrashenyi/50um-ladaf-2020-31-kidney-pag-0-01-0-02-jp2
```

## Setting up the environment 

```bash
# clone project
git clone https://github.com/burnmyletters/blood-vessel-segmentation
cd blood-vessel-segmentation

# [OPTIONAL] create conda environment
conda create -n bvs python=3.9
conda activate bvs

# install requirements
pip install -r requirements.txt
```

## How to run

Train model with default configuration

Note: don't forget to modify data_path in bash sripts and in generate_mutliview and generate_mutliview_pseudo files.

```bash
# generate multiview
cd scripts
python generate_mutliview.py

# train on base model
sh ./train.sh

# generate pseudos
cd src/utils
python generate_pseudo.py

# train on 3d model with pseudo
sh ./train_pseudo_3d.sh

# train on 2d model with pseudo
sh ./train_pseudo_v2.sh

# Note: if you want to train the best setup which was not submitted to the cometition run 
sh ./train_pseudo.sh

```
