import taichi as ti
import numpy as np
import os
import sys
import argparse

output_res = 1280, 720


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--res', type=int, help='Input resolution', default=512)
    parser.add_argument(
        '-d', '--dimension', type=int, help='Dimension', default=1)
    parser.add_argument('-i', '--indir', type=str, help='Input Folder')
    parser.add_argument('-o', '--outdir', type=str, help='Output folder')
    parser.add_argument(
        '--frames', type=int, help='Number of frames to render', default=0)
    parser.add_argument(
        '-v',
        '--visualize',
        action='store_true',
        help='Show gui when rendering')
    args = parser.parse_args()
    print(f'cmd args: {args}')
    return args


class DataType(object):
    GRAY_SCALE = 1
    RGB = 2


@ti.data_oriented
class SmokeRenderer:
    def __init__(self, res, data_type=DataType.GRAY_SCALE):
        self.res = res
        self.data_type = data_type
        if data_type == DataType.GRAY_SCALE:
            self.ray_traced = ti.field(dtype=ti.f32, shape=output_res)
            self.density = ti.field(dtype=ti.f32, shape=(res, res, res))
        elif data_type == DataType.RGB:
            self.ray_traced = ti.Vector.field(
                3, dtype=ti.f32, shape=output_res)
            self.density = ti.Vector.field(
                3, dtype=ti.f32, shape=(res, res, res))

        self.inv_dx = res
        self.dx = 1 / self.inv_dx
        self.density_scale = 100 / 255
        self.brightness = ti.field(dtype=ti.f32, shape=(res, res, res))
        self.light_dir = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_dir.from_numpy(np.array([1., 1., 1.]))
        # self.light_dir = self.light_dir.normalized()

    def fetch_ray_traced(self):
        assert self.dim == 3
        self.compute_brightness(self.dye[0])
        self.integrate(self.dye[0])
        return self.ray_traced.to_numpy()

    @ti.kernel
    def integrate(self):
        # camera_position = ti.Vector([0.4, 0.5, 0.7])
        camera_position = ti.Vector([0.25, 0.5, 0.9])
        fov = 3
        for P in ti.grouped(self.ray_traced):
            o = camera_position
            screen_coord = (
                P / self.ray_traced.shape[1] -
                0.5 * ti.Vector([output_res[0] / output_res[1], 1.0])) * fov
            d = (ti.Vector([screen_coord[0], screen_coord[1],
                            -1])).normalized()
            step_size = self.dx
            if ti.static(self.data_type == DataType.GRAY_SCALE):
                importance = 1.0
                contribution = 0.0
                for k in range(self.res * 3):
                    x = o + d * k * step_size
                    I = int(ti.floor(x * self.inv_dx))
                    if 0 <= I.min() and I.max() < self.res:
                        density = self.density[I] * self.dx * 2
                        importance *= ti.exp(-density)
                        contribution += self.brightness[
                            I] * importance * density
                    else:
                        break
                self.ray_traced[P] = importance + contribution
            else:
                importance = ti.Vector([1., 1., 1.])
                contribution = ti.Vector([0., 0., 0.])
                for k in range(self.res * 3):
                    x = o + d * k * step_size
                    I = int(ti.floor(x * self.inv_dx))
                    if 0 <= I.min() and I.max() < self.res:
                        for j in ti.static(range(3)):
                            density = self.density[I][j] * self.dx * 2
                            importance[j] = ti.exp(-density) * importance[j]
                            contribution[j] += self.brightness[I] * importance[
                                j] * density
                    else:
                        break
                # Assuming white backgrouund which has brightness importance
                self.ray_traced[P] = importance + contribution
                # self.ray_traced[P] = contribution

    @ti.kernel
    def compute_brightness(self):
        for J in ti.grouped(ti.ndrange(self.res, self.res)):
            occlusion = 0.0
            # for i_ in range(self.res):
            #     # i = self.res - 1 - i_
            #     i = i_
            #     density = 0.0
            #     if ti.static(self.data_type==DataType.GRAY_SCALE):
            #         density = self.density[i, J]
            #     else:
            #         density = self.density[i, J].norm()
            #     occlusion += density
            #     self.brightness[i, J] = ti.exp(-occlusion * self.dx)

            for i_ in range(self.res):
                i = self.res - 1 - i_
                # i = i_
                density = 0.0
                if ti.static(self.data_type == DataType.GRAY_SCALE):
                    density = self.density[J[0], i, J[1]]
                else:
                    density = self.density[J[0], i, J[1]].norm()
                occlusion += density
                self.brightness[J[0], i, J[1]] = ti.exp(-occlusion * self.dx)

    @ti.kernel
    def _compute_brightness(self):
        for i, j, k in ti.ndrange(self.res - 1, self.res - 1, self.res - 1):
            occlusion = 0.0
            I = ti.Vector(
                [self.res - 1 - i, self.res - 1 - j, self.res - 1 - k])
            for l in range(self.res * 3):
                next_cell = I + ti.Vector([-1, -1, -1]) * l
                if 0 <= next_cell.min() and next_cell.max() <= self.res:
                    if self.brightness[next_cell] == 0:
                        density = self.density[next_cell].norm()
                        occlusion += density
                        self.brightness[next_cell] = ti.exp(
                            -occlusion * self.dx)
                else:
                    break

    def load(self, fn):
        density = np.fromfile(fn, np.uint8)
        res = self.res
        if self.data_type == DataType.GRAY_SCALE:
            density = density.reshape(res, res, res)
        elif self.data_type == DataType.RGB:
            density = density.reshape(res, res, res, 3)
        self.density.from_numpy(
            density.astype(np.float32) * self.density_scale)

    def render(self):
        self.compute_brightness()
        # self._compute_brightness()
        self.integrate()
        return self.ray_traced.to_numpy()


if __name__ == '__main__':
    ti.init(arch=ti.gpu, device_memory_fraction=0.7)
    args = parse_args()
    res = args.res
    data_type = DataType.GRAY_SCALE
    if args.dimension == 3:
        data_type = DataType.RGB
    renderer = SmokeRenderer(res, data_type)
    show_gui = args.visualize
    # folder = f'sim_3d_{res}'  #sys.argv[0]

    folder = args.indir
    assert os.path.exists(folder), 'Please set correct input folder.'
    outdir = args.outdir
    if outdir == None:
        outdir = folder
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    frames = args.frames
    f_list = os.listdir(folder)
    for f in f_list:
        if str(os.path.splitext(f)[-1]) is '.np':
            frames += 1
    gui = ti.GUI('Smoke Renderer', output_res, show_gui=show_gui)
    for f in range(0, frames):
        print('frames: {}'.format(f))
        density_fn = os.path.join(folder, f'{f:04d}.np')
        output_fn = os.path.join(outdir, f'ray_traced_{f:04d}.png')
        renderer.load(density_fn)
        gui.set_image(renderer.render())
        gui.show(output_fn)
