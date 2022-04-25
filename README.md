# FewM-HGCL

## Installation

### Requirements

- Linux with Python ≥ 3.6
- [PyTorch ≥ 1.4.0](https://pytorch.org/)
- [0.5 > DGL ≥ 0.4.3](https://www.dgl.ai/pages/start.html)
- `pip install -r requirements.txt`
- Install [RDKit](https://www.rdkit.org/docs/Install.html) with `conda install -c conda-forge rdkit=2019.09.2`.

## Quick Start

1. build graphs:

```bash
python prepare/prepare.py
```
2. train:

```bash
cd GCC
sh run.sh 
```

[comment]: <> (```bash)

[comment]: <> (# run comparative experiment)

[comment]: <> (sh run_loop.sh)

[comment]: <> (```)

## Introduction to project structure

- utl some reusability features
- test some test demos, to test the linkability of the database, whether the file exists, and a simple example test
- train
  - prepare_gcc_data
  - GCC : gcc as a model for testing
- data (for the data part of the data, only the input part is temporarily uploaded, and the others need to be declared in gitignore)
  - input_data: input data
  - mid_data: data generated in the middle
  - out_data: output data
- prepare data preprocessing
  - read_data read data
  - draw_graph
  - build_dgl_from_graph
  - prepare.py
- analyze statistical work
- Materials Paper version of experimental materials


[comment]: <> (# Dataset Description)

[comment]: <> (布谷鸟数据集整理后的放在了mongoDB中。)

[comment]: <> (cuckoo_nfs_dX中的数据里，calls这个collection不是和analysis以及其他对应的。)
