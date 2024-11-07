# Environment setup 

Follow these steps to set up a `conda` environment named `causalign` with Python 3.11.10 and install dependencies from `requirements.txt`.

```bash
conda create -n causalign python=3.11.10
conda activate causalign
pip install -r requirements.txt
```

Then make sure your python version and packages are as expected. There may be some additional required packages for running some scripts. If you run into need for additional installs let me know so I can keep requirements.txt up to date if I have forgotten anything. 

```bash 
python --version
python -m pip list
```

# How to contribute

1. Make a fork of the repo
2. Make a branch in your fork named either f/feature_name or b/bug_name. Set upstream to this main repo, and origin to your fork. 
3. Regularly pull from upstream so your branch is up to date. Push to origin when you have a big push ready to go. When the bug is fixed or feature is complete, open a PR to upstream. 

^ Will add details later to make this easier. 

# `causalign` package structure

`causalign` contains the various scripts for causally regularized text matching:

`datasets/generators` contains preprocessing scripts to produce the train/val/test data (for ACL abstracts)

^ TODO: turn the preprocessing notebook into a script for replicability

`modules` contains the python file defining our pretrained BERT model. Specifically, `bert_pretrained.py` 
contains the logic for setting up a pretrained encoder model with a modern high-performance 
version of BERT for encoding (TBD, either use 'msmarco-distilbert-base-v3' like the OG Bansal 2023
paper, or a better newer BERT encoding model)

^ TODO: Implement this. Bansal's code could actually be useful here. 

`optim` contains the logic for ITVReg (intervention-based causal regularization), an
optimizer inheriting from AdamW but with additional logic for regularizing changes in 
word-level causal effects compared to the OG pretrained encoder. 

Proposed algorithm for ITVReg since Bansal did not share a clean implementation/ specific math: 
1. Compute word-by-word 'causal effects'. Computed by intervening (masking the word) and then 
computing $1-(masked encoding)^T(unmasked encoding) for each word.
2. Take absolute value of the difference of each word's causal effect in the current iteration
vs. in the baseline model (cache the baseline model causal effects to save memory). 
3. Sum the differences from (2), divide by number of examples, multiply by some hyperparameter lambda 
(to control strength of the regularization). 
4. Add the above regularization term to our loss. Might need to modify alg slightly to make sure the
regularization is differentiable (esp. abs value might need to become an L2 norm or similar instead)

^TODO: Implement this

`train.py` is the overall controller script for running encoder training. This will include an argument parser 
and is intended to be invoked from the CLI. We will likely need to train on CUDA, but ideally 
should try to make a script that is also compatible with MPS on Mac and then dynamically switch between machines 
(Mac for testing, GCP/Lambda for actual training) to save $$$. 

^TODO: Implement this, find interesting hyperparameters we should tweak, etc. 

# `notebooks` and other stuff

In `notebooks` there are currently three notebooks. 

`eda.ipynb` contains the data preprocessing logic 
for creating positive and negative examples for ACL abstract matching (based on 'citing' paper 
citing the 'cited' paper = positive example), starting from the ACL huggingface datasets. 

`experiment.ipynb` is where I plan to invoke the causalign package and perform some unit testing etc. 

`bansal-itvreg.ipynb` was a few hour effort where I attempted to rework Bansal's code (I scoured github and found it, 
but it is quite the mess). This reworking was unsuccessful and the code is messy and undocumented. I figure that re-implementing everything
will actually be a better open-source contribution + better scientific replication. 

...that's basically it for now. :)




