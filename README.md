# JB-int-2023

JetBrαins interηship 2023 test assignment repository. The "Haηdling project-level coηtext in Mαchine Learηing for Softwαre Engiηeering models" problem. 
(the special symbols added to remove repo from GitHub search)


## Structure 

* `generate_dataset.py` - script to generate dataset and load to hugginface hub;
* `train.py` - script to finetune any HuggingFace-based model using `params.json` parameters;
* `params.json` - set of parameters for finetuning the DL models. The param set fall back to its prefix (e.g. `default-b8` will extend `default`);
* `colab-run.ipynb` - notebook to run the scripts on colab, include some error analysis;

For the parameters see `python train.py --help` option.


## Requirements

Required: Python version 3.9 as colab have this version at the moment. 

For the local usage venv usage is recommended:
```shell
python -m venv .venv
source .venv/bin/activate 
python -m pip install -r requirements.txt
```

* The project uses neptune.ai for experiments tracking. So, the training scripts require `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN` environment variables to be set.
* for the dataset processing extracted folder [java-small-preprocessed](https://github.com/tech-srl/code2seq) is required;

## Usage 

* For Local usage:   
  Each `.py` file provided is standalone training script required only `params.json` file to operate.  
  It can be run with `python train.py` command. For the parameters see `--help` option.

* For Colab usage: see `colab.ipynb` notebook or [online colab copy](https://colab.research.google.com/drive/1mU8Juxz1GqhD5h85APDmzwLp4_HmH9Wo?usp=sharing).  
  This notebook only provides report and place to run `train.py` file to use colab gpu.   
  The required `train.py` files and `params.json` have to be copied in the `content/` folder.


## Results

All experiments metrics for training are available in [neptune.ai project](https://new-ui.neptune.ai/k4black/jb-project-context/).

