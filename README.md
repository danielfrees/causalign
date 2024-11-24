# CausalSent
Extension of Bansal et al. (2023)'s work using Riesz Representer learning to estimate
average treatment effect of words on sentiment outputs. Regularization of 
sentiment modeling based on adherence to running average ATE. Architecture optimizations
+ modularization + realistic data augmentation are a few of the critical extensions 
of the original experimental work. 

This repo is very much in development, but should be robustly implemented for future use 
as long as data preparation is implemented for new datasets similarly to IMDB/ 
CivilComments. 

To run the current (quick, limited data, overfitting) experiment, run: 

```bash
$ ./train_causal_sent.sh
```






# CausAlign (deprecated)
Research extending Bansal et al. (2023) [https://openreview.net/forum?id=5cCX_xZSeEl] for long text matching, such as ACL citation suggestions via abstract matching. Similarly extends Pang et al. (2021) [/arxiv.org/pdf/2101.06423] 

# data for similarity task (deprecated)
To use the embedding/similarity learning, first download the datasets `acl_full_citations.parquet` and `acl-publication-info.74k.v2.parquet` from https://huggingface.co/datasets/WINGNUS/ACL-OCL/tree/main into an `acl_data/` folder. These are large files so they are not pushed to the git repo. 