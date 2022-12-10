import argparse
import os


def params():
    parser = argparse.ArgumentParser()

    # File Options
    parser.add_argument(
        "--input-dir",
        default="assets",
        type=str,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        type=str,
        help="Output directory where the model predictions and checkpoints are written.",
    )
    parser.add_argument(
        "--grid-search-output",
        type=str,
        default='grid_search_output.npy',
    )
    parser.add_argument(
        "--dataset", default="amazon", type=str, help="dataset", choices=["amazon"]
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Whether to ignore cache and create a new input data",
    )

    # pytorch options
    parser.add_argument("--deterministic", action="store_true", help="use with --seed to set the randomizer seed for deterministic results.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--device",
        default=None,
        help="Which cuda device to use",
    )

    # Hyperparameters
    parser.add_argument(
        "--n-epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Batch size per GPU/CPU for training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        default=5e-5,
        type=float,
        help="Model learning rate starting point.",
    )
    parser.add_argument(
        "--evolution-epochs",
        default=-1,
        type=int,
        help="Number of epochs where evolution occurs, default to -1, where all epochs have evolution",
    )

    # transformer arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dense",
        action="store_true",
        help="Use dense transformer and classifier.",
    )
    group.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse transformer and classifier.",
    )
    parser.add_argument(
        "--transformer",
        default='sparse',
        choices=['sparse', 'dense'],
        help="Model to train.",
    )
    parser.add_argument(
        "--max-len", default=20, type=int, help="maximum sequence length to look back"
    )
    parser.add_argument(
        "--embed-dim",
        default=512,
        type=int,
        help="The embedding dimension of the transformer layers.",
    )
    parser.add_argument(
        "--feedforward-dim",
        default=2048,
        type=int,
        help="The feed forward dimension of the transformer layers.",
    )
    parser.add_argument(
        "--nhead",
        default=8,
        type=int,
        help="Number of attention heads in the transformer model.",
    )
    parser.add_argument(
        "--transformer_drop_rate",
        default=0.1,
        type=float,
        help="Dropout prob for transformer layers.",
    )
    parser.add_argument(
        "--num-transformer-layers",
        default=6,
        type=int,
        help="Number of transformer layers in the model."
    )
    parser.add_argument(
        "--sparse-layer-pweight",
        default=0.05,
        type=float,
        help="Controls number of connections in sparse layers, higher = more connections.",
    )
    parser.add_argument(
        "--sparse-layer-zeta",
        default=0.2,
        type=float,
        help="Fraction of connections pruned and replaced in sparse layers during each evolution step.",
    )

    # classifier
    parser.add_argument(
        "--classifier",
        default='sparse',
        choices=['sparse', 'dense'],
        help="Model to train.",
    )
    parser.add_argument(
        "--classifier-hidden-dim", default=1024, type=int, help="Model hidden dimension."
    )
    parser.add_argument(
        "--classifier-drop-rate", default=0.1, type=float, help="Dropout rate for classifier model."
    )
    parser.add_argument(
        "--classifier-pweight",
        default=0.1,
        type=float,
        help="Controls number of connections in sparse layers, higher = more connections.",
    )
    parser.add_argument(
        "--classifier-zeta",
        default=0.2,
        type=float,
        help="Fraction of connections pruned and replaced in sparse layers during each evolution step.",
    )

    args = parser.parse_args()
    if args.sparse:
        args.transformer = "sparse"
        args.classifier = "sparse"
    if args.dense:
        args.transformer = "dense"
        args.classifier = "dense"
    return args

