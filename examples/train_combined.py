"""Train a combined RNA/ADT H5AD: python train_combined.py input.h5ad output."""
import argparse
from citepool.model import CITEPool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_h5ad')
    parser.add_argument('output_dir')
    parser.add_argument('--resolution', type=float, default=1.0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    model = CITEPool(args.input_h5ad, initial_resolution=args.resolution)
    model.train(args.output_dir, device=args.device)
    print(model)
    print(model.predict().value_counts())


if __name__ == '__main__': main()
