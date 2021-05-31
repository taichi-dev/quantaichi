import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-type', type=int, help='data type of `dey density`: 0 for float32, 1 for shared exponent, 2 for fixed-point and 3 non-shared exponent', default=0)
    parser.add_argument(
        '-o', '--output', type=str, help='Output direction', default='outputs')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.data_type == 1 or args.data_type == 2 or args.data_type == 3:
        cmd = f'python demo.py -d 2 -r 1024 -o {args.output} --frame 300 --rk 3 --advect mc -q --demo-id 2 --dye-type {args.data_type-1}'
    elif args.data_type == 0:
        cmd = f'python demo.py -d 2 -r 1024 -o {args.output} --frame 300 --rk 3 --advect mc --demo-id 2'
    os.system(cmd)