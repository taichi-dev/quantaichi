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
        ci8 = ti.quant.int(8, True)

        x = ti.field(dtype=ci8)
        y = ti.field(dtype=ci8)
        z = ti.field(dtype=ci8)
        w = ti.field(dtype=ci8)

        ti.root.dense(ti.i, n).bit_struct(num_bits=32).place(x, y, z, w)
    else:
        x = ti.field(dtype=ti.i8)
        y = ti.field(dtype=ti.i8)
        z = ti.field(dtype=ti.i8)
        w = ti.field(dtype=ti.i8)

        ti.root.dense(ti.i, n).place(x, y, z, w)

    w[0] = 1

    @ti.kernel
    def partial_store():
        for i in range(n):
            x[i] = i & 127
            y[i] = i & 31
            z[i] = i & 7
            # do not store w[i]

    warmup_repeats = int(max(1, args.n * 0.2))
    for i in range(warmup_repeats):
        partial_store()

    ti.sync()
    ti.kernel_profiler_clear()
    t = time.time()

    for i in range(args.n):
        partial_store()

    ti.sync()
    print('total time:', time.time() - t)
    ti.kernel_profiler_print()
    assert w[0] == 1


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
