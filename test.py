import multiprocessing as mp
import argparse
import os
import yaml
import warnings
warnings.filterwarnings("ignore")
from tester import Tester


def main(args):
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config.items():
        setattr(args, k, v)

    # exp path
    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    # train
    tester = Tester(args)
    tester.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch De-Occlusion.')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--load_iter', default=None, type=int)
    parser.add_argument('--load-pretrain', default=None, type=str)
    args = parser.parse_args()

    main(args)
