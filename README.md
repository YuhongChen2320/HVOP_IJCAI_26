# HVOP

Code for **Boosting Knowledge Transfer and Retention with Brain-inspired Multi-View Incremental Learning**.

Published in **IJCAI 2026, Main Track**.

[Paper and publication details](https://ijcai.org/proceedings/2026/399)

## Citation

```bibtex
@inproceedings{chen2026hvop,
  title     = {Boosting Knowledge Transfer and Retention with Brain-inspired Multi-View Incremental Learning},
  author    = {Chen, Yuhong and Fang, Zihan and Yin, Huifeng and Wu, Yujie and Xu, Qi and Deng, Lei and Wang, Shiping and Xu, Mingkun},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence},
  year      = {2026},
  pages     = {3585--3593},
  doi       = {10.24963/ijcai.2026/399},
  url       = {https://ijcai.org/proceedings/2026/399}
}
```

## Environment

| Dependency | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| scikit-learn | 1.5.2 |
| Matplotlib | 3.9.4 |
| tqdm | 4.67.1 |
| texttable | 1.7.0 |

The tested environment is available at `/data2/chyh/.venvs/GNN`. The complete dependency snapshot is provided in [results/environment.lock.txt](results/environment.lock.txt). The verified execution device is CPU.

## Running

The Flower17 dataset and its seven precomputed adjacency matrices are included:

```text
datasets/flower17.mat
adj_matrix/flower17/v{0..6}_knn15_adj.npz
```

Run from the project directory:

```bash
cd /data2/chyh/ICLR_code_HVOP

# Smoke run: one epoch per view
./run_flower17.sh --num_epoch 1 > results/flower17_smoke_new.log 2>&1

# Full run: 800 epochs per view
./run_flower17.sh > results/flower17_full_new.log 2>&1
```

The script uses the tested Python environment and defaults to CPU. Update the Python path in `run_flower17.sh` when using another environment. Logs are saved through shell redirection.

Alternatively, invoke the training entry point directly:

```bash
python -u train.py --device cpu --dataset flower17 --num_epoch 800
```

Use `--device 0` to request CUDA device 0 when a working GPU environment is available.

## Parameter Settings

The default Flower17 configuration is defined in `args.py`.

| Parameter | Default | Description |
| --- | --- | --- |
| `dataset` | `flower17` | Dataset name |
| `path` | `./datasets/` | Dataset directory |
| `device` | `0` | CUDA device index, or `cpu`; the supplied script selects CPU |
| `num_epoch` | `800` | Training epochs per view; 5600 epochs across seven views |
| `hidden` | `256` | Hidden dimension |
| `lr` | `0.006` | Adam learning rate |
| `weight_decay` | `0.0005` | Weight decay |
| `ratio` | `0.1` | Labeled sample ratio |
| `seed` | `42` | Model initialization seed |
| `sf_seed` | `2021` | Label partition seed |
| `knns` | `15` | Neighbors used by the supplied Flower17 graphs |
| `n_repeated` | `1` | Number of runs |

Views are processed in the order `0, 1, 2, 3, 4, 5, 6`. Flower17 contains 1360 samples and 17 classes. The default partition uses 136 labeled samples and 1224 unlabeled samples for evaluation.

Additional arguments are listed by:

```bash
python train.py --help
```

Example with custom settings:

```bash
./run_flower17.sh --num_epoch 100 --lr 0.006 --hidden 256 --seed 42
```

## File Description

| File / Directory | Description |
| --- | --- |
| `train.py` | Sequential multi-view training and evaluation |
| `args.py` | Command-line arguments and default settings |
| `Dataloader.py` | Dataset loading, label partitioning, and adjacency normalization |
| `model.py` | Model architecture |
| `hlop_module.py` | Hebbian subspace learning and orthogonal projection |
| `proj_linear.py` | Linear layers with gradient projection |
| `utils.py` | Evaluation metrics and utility functions |
| `run_flower17.sh` | CPU execution script for Flower17 |
| `datasets/` | Dataset files |
| `adj_matrix/` | Precomputed adjacency matrices |
| `results/` | Training logs, environment snapshot, and evaluation summaries |
