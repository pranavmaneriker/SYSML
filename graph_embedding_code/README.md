# Create graph embeddings

This folder contains codes for generating context embeddings.

## Make

Use this command to compile the codes. Make sure to change the compiler and other parameters in `makefile` if you need.

`make
`


## How to use

### Data construction

`./dataConstruction -rawdata <file> -outputdir <dir> -padding <int> -msglist
`

- `-rawdata`: Path of your raw data file.
- `-outputdir`: Directory in which the preprocessed data will be stored
- `-padding`: Padding of node ids between different types of nodes. Be sure to use values larger than the size of any type of nodes.
- `-msglist`: optional, whether to output the id-msg mapping.


### Generate random walk

`./generateWalks -datadir <dir> -numwalks <int> -walklength <int>
`

- `-datadir`: Directory of preprocessed data.
- `-numwalks`: Number of walks generated for each node.
- `-walklength`: Length of each random walk.


### Heterogeneous graph embedding

`./metapath2vec -train <file> -output <file> -pp <int> -size <int> -window <int> -negative <int> -threads <int>
`

- `-train`: Path of the random walk file.
- `-output`: Path of output embedding file.
- `-pp`: Use metapath2vec++ (1) or metapath2vec (0). Default is 1.
- `-size`: Dimensionality of embeddings.
- `-window`: Window size for skip-gram model.
- `-negative`: Number of negative samples.
- `-threads`: Number of threads used.




### References

The code uses or adapted from following repos:

- CSV parser: https://github.com/AriaFallah/csv-parser
- pbar: https://github.com/Jvanrhijn/pbar
- metapath2vec: modified from https://ericdongyx.github.io/metapath2vec/m2v.html


### Example usage

We assume that the splits (`data/rasmus/cleaned/splits/`) are in `$DATA_DIR`
```
make dataConstruction
time ./dataConstruction -rawdata $DATA_DIR/sr2/train.csv -outputdir sr2 -padding 500000
make generateWalks
time ./generateWalks -datadir sr2/ -numwalks 200 -walklength 80
make metapath2vec
time ./metapath2vec -train sr2/walks_sr_200_80.txt -output sr2/embedding0526.txt -pp 1 -size 128 -window 7 -negative 5 -threads 32
```


