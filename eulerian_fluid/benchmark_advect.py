import taichi as ti
import time
import argparse

from solver import FluidSolverBase
from utils import tube_domain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dim', type=int, help='Dimension (2 or 3)', default=2)
    parser.add_argument(
        '-r', '--res', type=int, help='Resolution', default=512)
    parser.add_argument(
        '-D', '--debug', action='store_true', help='Use debug mode')
    parser.add_argument(
        '-a', '--async-mode', action='store_true', help='Use async mode')
    parser.add_argument(
        '-q',
        '--quant-all',
        action='store_true',
        help='Run in quantization mode which will quantize velocity and dye.')
    parser.add_argument(
        '-qv',
        '--quant-v',
        action='store_true',
        help='Quantize velocity field')
    parser.add_argument(
        '-qd',
        '--quant-dye',
        action='store_true',
        help='Quantize dye density field')
    parser.add_argument(
        '--dye-type',
        type=int,
        default=0,
        help=
        'quantized dype type: 0 for shared-exp, 1 for fixed-point and 2 for non-shared exponent'
    )
    parser.add_argument(
        '--no-gui', action='store_true', help='Do not show GUI')
    parser.add_argument('--dot', action='store_true', help='Export dot file')
    parser.add_argument(
        '--visualize', action='store_true', help='Visualize result')
    parser.add_argument(
        '--advect',
        type=str,
        help='Advection method (\'mc\': MacCormack; \'sl\': Semi-Lagrangian)',
        default='mc')
    parser.add_argument(
        '--frames',
        type=int,
        help=
        'Number of frames to run. We can set this to a smaller value during benchmarks',
        default=100000)
    parser.add_argument(
        '--benchmark-id', type=int, help='benchmark experiment index', default=0)
    args = parser.parse_args()
    print(f'cmd args: {args}')
    return args


@ti.data_oriented
class Benchmark(FluidSolverBase):
    def __init__(self,
                 dim,
                 res,
                 advect_op,
                 benchmark_id=0,
                 v_quant=False,
                 dye_quant=False,
                 dye_type=0):
        super().__init__(
            res=res,
            rk_order=3,
            advect_op=advect_op,
            dim=dim,
            demo_id=3,
            benchmark_id=benchmark_id,
            v_quant=v_quant,
            dye_quant=dye_quant,
            dye_type=dye_type)

        self.bound = []
        for i in range(self.dim):
            self.bound.append([self.offset[i], self.res + self.offset[i]])

        # |v| is originally a SparseFieldCollection. Since we don't advect the
        # velocity field, we just retain the first field in the collection.
        self.v = self.v[0]
        # for d in self.dye:
        #     d.create_decomposed_sparse_fields()

        self.pixels = ti.Vector.field(
            3, dtype=float, shape=(self.res, self.res))

    @ti.kernel
    def init(self):
        for I in ti.grouped(ti.ndrange(*self.bound)):
            # Use a tube domain to initialize the sparse grid
            xy = I / self.res
            if tube_domain(self.dim, xy):
                init_v = ti.Vector.zero(ti.f32, self.dim)
                # (-y, x, 0)
                init_v[0] = -xy[1] * 1.
                init_v[1] = xy[0] * 1.
                self.v.field[I] = init_v

                color = ti.sin(ti.Vector([xy[0], xy[1], xy[0] * xy[1]
                                          ])) * 0.5 + 0.5
                for d in ti.static(list(self.dye)):
                    d.field[I] = color

    def advect(self):
        # for i in range(3):
        #     self.advection_op.advect(
        #         self.v, self.dye[0].get_sparse_field_at(i),
        #         self.dye[1].get_sparse_field_at(i),
        #         self.dye[2].get_sparse_field_at(i), self.dt)
        self.advection_op.advect(self.v, self.dye[0], self.dye[1], self.dye[2],
                                 self.dt)
        self.dye.swap(0, 1)

    def paint(self):
        self._paint(self.dye[0].field)

    @ti.kernel
    def _paint(self, dye: ti.template()):
        for i, j in self.pixels:
            if ti.static(self.dim == 2):
                self.pixels[i, j] = dye[i + self.offset[0], j + self.offset[1]]
            else:
                self.pixels[i, j] = dye[i + self.offset[0], j + self.offset[1],
                                        0]


if __name__ == '__main__':
    args = parse_args()
    kwargs = {}
    if args.dot:
        kwargs['async_opt_intermediate_file'] = "dots/advect"
    ti.init(
        arch=ti.gpu,
        kernel_profiler=True,
        async_mode=args.async_mode,
        debug=args.debug,
        device_memory_fraction=0.8,
        **kwargs)
    v_quant = args.quant_v
    dye_quant = args.quant_dye
    dye_type = args.dye_type
    if args.quant_all:
        v_quant = True
        dye_quant = True
    solver = Benchmark(
        dim=args.dim,
        res=args.res,
        advect_op=args.advect,
        benchmark_id=args.benchmark_id,
        v_quant=v_quant,
        dye_quant=dye_quant,
        dye_type=dye_type)
    solver.init()
    show_gui = not args.no_gui
    if show_gui:
        gui = ti.GUI(res=args.res, show_gui=show_gui)
    second_frame_begin_ts = None
    for f in range(args.frames):
        solver.advect()
        if show_gui and args.visualize:
            solver.paint()
            gui.set_image(solver.pixels)
            gui.show()
        if f == 0:
            second_frame_begin_ts = time.time()
        ti.sync()
        if f % 10 == 0:
            print(f'Ran {f} frames', flush=True)
        if f == 1:
            # Exclude compilation time
            ti.profiler.clear_kernel_profiler_info()
    print(
        f'Advection benchmark time: {time.time() - second_frame_begin_ts:.3f} s'
    )
    #ti.misc.util.print_async_stats(include_kernel_profiler=True)
    ti.profiler.print_kernel_profiler_info()
