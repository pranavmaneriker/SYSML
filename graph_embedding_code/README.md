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



