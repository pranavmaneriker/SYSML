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

* `data`: See `data/README.md` (from `data_link.txt`)
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

However, the model has been tested and trains/runs successfully on a desktop with 16GB RAM, 512GB HDD.

### Runtime

Note that the first run takes a long time as the episode dataset is built. Subsequent runs use a cached version of the dataset; the cost is amortized.

Average running time: 27:17 per run on multitask model.

Num paramereters: The multitask model has 5189138 parameters (float32). This is the largest model used.

Hyperparams opimization was done by comparing average validation loss across 5 runs.

Hyperparams are reported in supplementary material.

### Notes

Data preprocessing scripts are provided with the data (see `data_link.txt`).


## Citation

If you ues our work, please use the following citation:

```
@inproceedings{maneriker-etal-2021-sysml,
    title = "{SYSML}: {S}t{Y}lometry with {S}tructure and {M}ultitask {L}earning: {I}mplications for {D}arknet Forum Migrant Analysis",
    author = "Maneriker, Pranav  and
      He, Yuntian  and
      Parthasarathy, Srinivasan",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.548",
    doi = "10.18653/v1/2021.emnlp-main.548",
    pages = "6844--6857",
    abstract = "Darknet market forums are frequently used to exchange illegal goods and services between parties who use encryption to conceal their identities. The Tor network is used to host these markets, which guarantees additional anonymization from IP and location tracking, making it challenging to link across malicious users using multiple accounts (sybils). Additionally, users migrate to new forums when one is closed further increasing the difficulty of linking users across multiple forums. We develop a novel stylometry-based multitask learning approach for natural language and model interactions using graph embeddings to construct low-dimensional representations of short episodes of user activity for authorship attribution. We provide a comprehensive evaluation of our methods across four different darknet forums demonstrating its efficacy over the state-of-the-art, with a lift of up to 2.5X on Mean Retrieval Rank and 2X on Recall@10.",
}
```
