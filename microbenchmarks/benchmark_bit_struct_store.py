import taichi as ti
import argparse
import time
import benchmark_utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-q',
                        '--quant',
                        action='store_true',
                        help='Use quantized type')

    benchmark_utils.add_benchmark_args(parser)

    args = parser.parse_args()
    print(f'cmd args: {args}')
    return args


def run(args):
    ti.init(**benchmark_utils.extract_init_kwargs(args))

    quant = args.quant

    n = 1024 * 1024 * 256

    if quant:
        ci16 = ti.quant.int(16, True)

        x = ti.field(dtype=ci16)
        y = ti.field(dtype=ci16)

        ti.root.dense(ti.i, n).bit_struct(num_bits=32).place(x, y)
    else:
        x = ti.field(dtype=ti.i16)
        y = ti.field(dtype=ti.i16)

        ti.root.dense(ti.i, n).place(x, y)

    @ti.kernel
    def store():
        for i in range(n):
            x[i & 32767] = i & 1023
            y[i & 32767] = i & 15

    @ti.kernel
    def check():
        for i in range(32768):
            assert x[i] == i & 1023
            assert y[i] == i & 15

    warmup_repeats = int(max(1, args.n * 0.2))
    for i in range(warmup_repeats):
        store()

    ti.sync()
    ti.kernel_profiler_clear()
    t = time.time()

    for i in range(args.n):
        store()

    ti.sync()
    print('total time:', time.time() - t)
    ti.kernel_profiler_print()

    check()


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
