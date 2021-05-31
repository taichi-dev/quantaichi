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
    ti.init(device_memory_GB=2, **benchmark_utils.extract_init_kwargs(args))

    quant = args.quant

    n = 1024 * 1024 * 32

    if quant:
        F_bound = 4.0
        cft = ti.quant.fixed(frac=16, range=(F_bound + 0.1))
        F = ti.Matrix.field(3, 3, dtype=cft)
    else:
        F = ti.Matrix.field(3, 3, dtype=ti.f32)

    # block = ti.root.dynamic(ti.i, 2 ** 30, 2 ** 25)
    block = ti.root.dense(ti.i, n)

    if quant:
        block.bit_struct(num_bits=32).place(F(0, 0), F(0, 1))
        block.bit_struct(num_bits=32).place(F(0, 2), F(1, 0))
        block.bit_struct(num_bits=32).place(F(1, 1), F(1, 2))
        block.bit_struct(num_bits=32).place(F(2, 0), F(2, 1))
        block.bit_struct(num_bits=32).place(F(2, 2))
    else:
        block.place(F)

    @ti.kernel
    def matmul():
        for i in range(n):
            F[i] = (ti.Matrix.identity(ti.f32, 3) +
                    0.1 * ti.Matrix.one(ti.f32, 3, 3)) @ F[i]

    warmup_repeats = int(max(1, args.n * 0.2))
    for i in range(warmup_repeats):
        matmul()

    ti.sync()
    ti.kernel_profiler_clear()
    t = time.time()

    for i in range(args.n):
        matmul()

    ti.sync()
    print('total time:', time.time() - t)
    ti.kernel_profiler_print()


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
