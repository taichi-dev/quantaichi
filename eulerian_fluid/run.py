import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--demo', type=int, help='Demo id', default=0)
    parser.add_argument(
        '-o', '--output', type=str, help='Output direction', default='outputs')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.demo == 0:
        os.system('tar zxvf inputs/640.np.tar.gz -C inputs')
        cmd = f'python demo.py -d 3 -r 768 -o {args.output} --frame 300 --rk 3 --advect mc -q --demo-id {args.demo}'
    elif args.demo == 1:
        cmd = f'python demo.py -d 3 -r 896 -o {args.output} --frame 280 --rk 3 --advect mc -q --demo-id {args.demo}'
    os.system(cmd)