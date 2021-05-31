import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--experiment', type=int, help='experiment setting: 0 for all float32, 1 for shared_exponent velocity(5 exp + 9 fractions) and fixed 10 for dy density, 2 for all fixed 10', default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.experiment == 0:
        cmd = 'python benchmark_advect.py -d 3 -r 256 --frames 1000 --no-gui --benchmark-id 0'
    elif args.experiment == 1 or args.experiment == 2:
        cmd = f'python benchmark_advect.py -d 3 -r 256 --frames 1000 -q --dye-type 1 --no-gui --benchmark-id {args.experiment}'
    os.system(cmd)