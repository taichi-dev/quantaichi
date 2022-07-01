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
        qflt = ti.types.quant.float(frac=22, exp=8)
        x = ti.field(dtype=qflt)
        y = ti.field(dtype=qflt)

        ti.root.dense(ti.i, n).bit_struct(num_bits=32).place(x)
        ti.root.dense(ti.i, n).bit_struct(num_bits=32).place(y)
    else:
        x = ti.field(dtype=ti.f32)
        y = ti.field(dtype=ti.f32)

        ti.root.dense(ti.i, n).place(x)
        ti.root.dense(ti.i, n).place(y)

    @ti.kernel
    def saxpy(alpha: ti.f32):
        for i in range(n):
            y[i] = alpha * x[i] + y[i]

    warmup_repeats = int(max(1, args.n * 0.2))
    for i in range(warmup_repeats):
        saxpy(1.1)

    #ti.sync()
    ti.profiler.clear_kernel_profiler_info()
    t = time.time()

    for i in range(args.n):
        saxpy(1.1)

    #ti.sync()
    print('total time:', time.time() - t)
    ti.profiler.print_kernel_profiler_info()


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
