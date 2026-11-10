
import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parser.add_argument("--path", type=str, default="./datasets/", help="Path of datasets")
    parser.add_argument("--dataset", type=str, default="flower17", help="Name of datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--sf_seed", type=int, default=2021, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")

    parser.add_argument("--n_repeated", type=int, default=1, help="Number of repeated times. Default is 10.")
    parser.add_argument("--save_results", action='store_true', default=True, help="xx")
    parser.add_argument("--save_all", action='store_true', default=False, help="xx")
    parser.add_argument("--save_loss", action='store_true', default=True, help="xx")
    parser.add_argument("--save_ACC", action='store_true', default=True, help="xx")
    parser.add_argument("--save_F1", action='store_true', default=True, help="xx")

    parser.add_argument("--knns", type=int, default=15, help="Number of k nearest neighbors")
    parser.add_argument("--common_neighbors", type=int, default=2, help="Number of common neighbors (when using pruning strategy 2)")
    parser.add_argument("--pr1", action='store_true', default=True, help="Using prunning strategy 1 or not")
    parser.add_argument("--pr2", action='store_true', default=True, help="Using prunning strategy 2 or not")
    parser.add_argument("--ghost", action='store_true', default=False, help="xx")


    parser.add_argument("--lr", type=float, default=0.006, help="flower17")
    parser.add_argument("--hidden", type=int, default=256, help="flower17")
    parser.add_argument("--a", type=float, default=0.178, help="flower17")
    # parser.add_argument("--lr", type=float, default=0.01929, help="YB")
    # parser.add_argument("--hidden", type=int, default=512, help="YB")
    # parser.add_argument("--a", type=float, default=0.0143, help="YB")
    parser.add_argument("--b", type=float, default=1, help="flower17")
    parser.add_argument("--weight_decay", type=float, default= 5e-4, help="Weight decay")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio of labeled samples")
    parser.add_argument("--num_epoch", type=int, default=800, help="Number of training epochs. Default is 200.")
    parser.add_argument("--alpha", type=float, default=5, help="beta")
    parser.add_argument("--rho", type=float, default=0.05, help="rho")
    parser.add_argument("--dim1", type=int, default=4, help="Number of hidden dimensions")
    parser.add_argument("--dim2", type=int, default=4, help="Number of hidden dimensions")

    args = parser.parse_args()


    return args