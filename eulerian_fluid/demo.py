from solver import *
import argparse
import time
import sys
import os
import os.path
import numpy

show_velocity = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dim', type=int, help='Dimension (2 or 3)', default=2)
    parser.add_argument(
        '-r', '--res', type=int, help='Resolution', default=512)
    parser.add_argument('--rk', type=int, help='RK order (1, 2, 3)', default=3)
    parser.add_argument(
        '--advect',
        type=str,
        help='Advection method (\'mc\': MacCormack; \'sl\': Semi-Lagrangian)',
        default='mc')
    # Cannot use '--async' because it's a keyword...
    parser.add_argument(
        '-D', '--debug', action='store_true', help='Use debug mode')
    parser.add_argument(
        '-a', '--async-mode', action='store_true', help='Use async mode')
    parser.add_argument(
        '-v', '--visualize', action='store_true', help='Visualize result')
    parser.add_argument(
        '--demo-id', type=int, help='Demo id', default=0)
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
        '-o',
        '--outdir',
        type=str,
        help=
        'Output folder. If specified, the output image sequence (and density, if 3D) will be written to \'$outdir_$res/\''
    )
    parser.add_argument(
        '--frames',
        type=int,
        help=
        'Number of frames to run. We can set this to a smaller value during benchmarks',
        default=10000)
    args = parser.parse_args()
    print(f'cmd args: {args}')
    return args


def fetch_colors():
    # dyes = solver.fetch_vector_to_numpy(solver.dye[0])
    # dyes = solver.fetch_color()
    dyes = solver.fetch_color_by_slice()
    # activation = solver.fetch_grid_activation_slice() * 0.1
    # broadcast to match |dyes|'s dimension
    # activation = activation[..., np.newaxis]
    # colors = dyes + activation
    colors = dyes
    return colors


if __name__ == '__main__':
    # ti.metal has some trouble handling the offsets...
    args = parse_args()
    ti.init(
        arch=ti.cuda,
        debug=args.debug,
        kernel_profiler=True,
        async_mode=args.async_mode,
        device_memory_fraction=0.9)

    res = args.res
    dim = args.dim
    is_3d = dim == 3
    v_quant = args.quant_v
    dye_quant = args.quant_dye
    if args.quant_all:
        v_quant = True
        dye_quant = True
    solver = FluidSolver(
        res=res,
        rk_order=args.rk,
        advect_op=args.advect,
        reflection=True,
        source_radius=0.0625 if is_3d else 0.01,
        dim=dim,
        demo_id=args.demo_id,
        v_quant=v_quant,
        dye_quant=dye_quant,
        dye_type=args.dye_type)
    pic_output_folder = None
    np_output_folder = None
    if args.outdir:
        pic_output_folder = f'{args.outdir}/pic/'
        np_output_folder = f'{args.outdir}/np/'
        os.makedirs(pic_output_folder, exist_ok=True)
        os.makedirs(np_output_folder, exist_ok=True)
    show_gui = args.visualize
    gui_res = res
    gui = ti.GUI('Taichi fluids', (gui_res, gui_res), show_gui=show_gui)
    for f in range(args.frames):
        t = time.time()
        gui.set_image(fetch_colors())
        img_path = None
        if pic_output_folder:
            img_path = os.path.join(pic_output_folder, f'{f:04d}.png')
        if np_output_folder:
            density_path = os.path.join(np_output_folder, f'{f:04d}.np')
            if is_3d:
                solver.dump_density(density_path)
        gui.show(img_path)
        try:
            ti.profiler.print_memory_profiler_info()
        except:
            pass
        solver.step()
        print(f'Frame {f} time: {(time.time() - t) * 1000:.3f}ms')
        print(
            f'Activated Voxels: {solver.count_activated_voxels()}', flush=True)
        if args.demo_id == 2:
            print(f'Density sum: {solver.sum_density(solver.dye[0].field):.3f}')
    # ti.print_profile_info()
    # ti.misc.util.print_async_stats(include_kernel_profiler=True)
