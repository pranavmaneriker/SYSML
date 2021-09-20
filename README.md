SYSML: StYlometry with Structure and Multitask Learning:Implications for Darknet Forum Migrant Analysis
=======================================================================================================

Code and Data for the paper "SYSML: StYlometry with Structure and Multitask Learning: Implications for Darknet Forum Migrant Analysis" accepted at EMNLP 2021

### Setup

* Install python (tested with python 3.7.3)
* Setup a virtualenv and activate it
```
virtualenv venv
source venv/bin/activate
```
* Install dependencies
```
pip install -r requirements.txt
```

### Running

For single dataset model, see `scripts/single_model.sh`

For multitask model, see `scripts/multitask_model.sh`

### Structure

Note: Please place data folder in this dir as shown by structure or adjust script data paths accordingly.

* `data`: See `data/README.md` (from `data\_link`)
* `graph_embedding_code`: Generates hetereogeneous graph embeddings. See `graph_embedding_code/README.md`
* `README.md`: This file
* `requirements.txt`: pip install package list
* `scripts`: Instructions for running the proposed model on single and multiple markets.
* `src`: Implementation of SYSML

### Environment

Experiments were run on SLURM cluster machines with config:

* OS: CentOS Linux release 7.8.2003 (Core)

* Hardware: 
    * CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
    * GPU: Tesla K80
    * RAM: 131734940 kB 
    * HDD: 7.3 T

However, the model runs on a desktop with ~16GB RAM, 512GB HDD as well.

### Runtime

Note that the first run takes a long time as the episode dataset is built. Subsequent runs use a cached version of the dataset the cost is amortized.

Average running time: 27:17 per run on multitask model.

Num paramereters: The multitask model has 5189138 parameters (float32). This is the largest model used.

Hyperparams opimization was done by comparing average validation loss across 5 runs.

Hyperparams are reported in supplementary material.

### Notes

Data preprocessing scripts are provided at the data link.

