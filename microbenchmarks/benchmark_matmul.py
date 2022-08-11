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

    n = 1024 * 1024 * 32

    if quant:
        F_bound = 4.0
        qfxt = ti.types.quant.fixed(bits=16, max_value=(F_bound + 0.1))
        F = ti.Matrix.field(3, 3, dtype=qfxt)
    else:
        F = ti.Matrix.field(3, 3, dtype=ti.f32)

    # block = ti.root.dynamic(ti.i, 2 ** 30, 2 ** 25)
    block = ti.root.dense(ti.i, n)

    if quant:
        bitpack = ti.BitpackedFields(max_num_bits=32)
        bitpack.place(F.get_scalar_field(0, 0), F.get_scalar_field(0, 1))
        block.place(bitpack)
        bitpack = ti.BitpackedFields(max_num_bits=32)
        bitpack.place(F.get_scalar_field(0, 2), F.get_scalar_field(1, 0))
        block.place(bitpack)
        bitpack = ti.BitpackedFields(max_num_bits=32)
        bitpack.place(F.get_scalar_field(1, 1), F.get_scalar_field(1, 2))
        block.place(bitpack)
        bitpack = ti.BitpackedFields(max_num_bits=32)
        bitpack.place(F.get_scalar_field(2, 0), F.get_scalar_field(2, 1))
        block.place(bitpack)
        bitpack = ti.BitpackedFields(max_num_bits=32)
        bitpack.place(F.get_scalar_field(2, 2))
        block.place(bitpack)
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

    #ti.sync()
    ti.profiler.clear_kernel_profiler_info()
    t = time.time()

    for i in range(args.n):
        matmul()

    #ti.sync()
    print('total time:', time.time() - t)
    ti.profiler.print_kernel_profiler_info()


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
