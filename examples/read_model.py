"""Read a trained workflow: python read_model.py /path/to/model."""
import argparse
from citepool.model import CITEPool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    model = CITEPool.load(args.directory)
    print(model)
    print('RNA latent:', model.get_latent_representation().shape)
    print('Protein predictions:', model.get_reconstructed_protein().shape)
    print(model.predict().value_counts())


if __name__ == '__main__': main()
