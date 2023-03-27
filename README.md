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

* For Colab usage: the scripts can we used as-is in the colab to get "free-gpu". 
  The required `train.py` files and `params.json` have to be copied in the `content/` folder.

Example:
```shell
python train.py --config-name=default --base-model=t5-small --push-to-hub 
```

## Report

All experiments metrics for training are available in [neptune.ai project](https://new-ui.neptune.ai/k4black/jb-project-context/).

### Dataset processing 

As the [original dataset](https://github.com/tech-srl/code2seq) has a bit (a lot) strange format, 
we've processed it and loaded it to the Huggingface Hub for stream-less use in the training. 

The processing script is `generate_dataset.py`. It requires `java-small-preprocessed` folder to be in the root of the project. 
The processed dataset was loaded as `k4black/code2seq-java-small-name-prediction` (private due to the license).

Each train/val/test file processed the following way:
* Method name is extracted to the `label` columns and remove separator:  
  e.g. `set|api|key|credentials` -> `set api key credentials`;
* Similarly, the body was cleared to reduce the number of tokens:  
  e.g. `Prim0|Mth|Nm1` -> `Prim0 Mth Nm1`;
* Body separator `,` was replaced with `ELEM_SEP` to detect it easily:  
  e.g. `boolean,Prim0|Mth|Nm1,METHOD_NAME boolean` -> `boolean ELEM_SEP Prim0|Mth|Nm1 ELEM_SEP METHOD_NAME boolean`;

### Model selection 

Due to limited resources we experimented only with `small`-sized models.

During solving another JB internship task (`Analysis of internal representations in code generation models`) 
[we experimented with different models and found](https://github.com/k4black/JB-internship-2023-internal-representations) than `Salesforce/codet5-small` performs better for code-generation task.  
However, in case, we experimented with `t5-small` and `flan-t5-model` and found... `codet5` is better =)


### Tokens split

With selected dataset we can check obtained tokens with the model tokenizer.  
As we have a lot of special tokens in the input (e.g. `Prim0`, `Mth`, `Nm1`) the tokenization show itself as a problem. 
It split the tokens in a lot of sub-tokens which should complicate the model training (e.g. `Prim0` -> `P, _rim, _0`).

Ideally (accounting quite large 'java-small' dataset) we can consider training tokenizer (so the model) from scratch. 
However, in out case it's not possible due to the limited resources.

So we decided to use the tokenizer as-is and try to finetune the model to work with it. Another option we considered is 
to add some tokens (most popular e.g. `Prim`, `Cal`, `Nm` etc.) to the existing tokenizer and train only these tokens.*

Note: *it did not work out for now.

### Metrics 

Looking at the various benchmarks the following frequently used metrics are highlighted:

* `exact_match` - exact match of generated code. 
* `rouge` - popular translation/summarication metric - indicated n-grams overlap between generated and target text 
* `bleu` - popular translation mertic - caclulate matching n-grams   
* `pass` (validation) - code generation metrics - depends on the number of passed unit-tests.

We don't have unittests to pass, so `exact_match` and `rouge` were selected to measure during training. 

### Model finetuning

Using `train.py` script we conducted a number of experiments.

In general, it is complicated dataset to train the model on. The main reasons we see is the following:
* A lot of special tokens in the input, additionally split by the tokenizer;
* A lot of meaning-less tokens e.g. `Nm` or `Bk`, so the model can not use the pretraining "experience";
* Highly structured input (e.g. `Prim0|Mth|Nm1`) which can be hard to learn for the model, which was trained on the natural code generation;
* Sometimes extremely long context truncated to the 512 tokens (`t5-small` input length);
* The dataset is quite large (700K samples) so the training process is quite long.

Experimenting with learning rate we found the model not training well, but the `lr=1e-4` works better (`5e-5` a bit better, but fluctuating a lot).

| learning rate | Exact Match | rouge1  |
|---------------|-------------|---------|
| 3e-4          | 0.2988      | 0.4930  |
| 1e-4          | 0.3049      | 0.5096  |
| 5e-5          | 0.3112      | 0.5091  |
| 1e-5          | 0.2849      | 0.4953  |

(batch size of 64)

Experimenting with batch size we found increasing batch size to be more stable in training, so maximum we can fit in gpu memory `batch_size=96` was selected.

| batch size | Exact Match | rouge1 |
|------------|-------------|--------|
| 32         | 0.3031      | 0.5204 |
| 64         | 0.3049      | 0.5096 |
| 96         | 0.3308      | 0.5296 |


The final model was trained with `batch_size=96` and `lr=1e-4` for 10 epochs available as [k4black/Salesforce-codet5-small-java-small-selected-wo-tokens](https://huggingface.co/k4black/Salesforce-codet5-small-java-small-selected-wo-tokens) Huggingface Hub.
the model achieved the following metrics: `exact_match=0.3302, rouge1=0.5329, rouge2=0.2582, rougeL=0.5323`