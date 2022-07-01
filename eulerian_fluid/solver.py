import taichi as ti
import math
import numpy as np

from mgpcg import MGPCG

from sparse_field import SparseField
from sparse_field_collection import SparseFieldCollection
from advection import AdvectionOp



class DyeType:
    SHARED_EXP = 0
    FIXED_POINT = 1
    NON_SHREAD_EXP = 2


class SmokeType:
    INIT_BY_SHAPE = 0
    FLOW = 1


@ti.data_oriented
class FluidSolverBase:
    def __init__(self,
                 res,
                 rk_order,
                 advect_op,
                 dim,
                 demo_id=0,
                 benchmark_id=0,
                 v_quant=False,
                 dye_quant=False,
                 dye_type=0):
        self.dim = dim
        self.res = res
        self.demo_id = demo_id
        self.benchmark_id = benchmark_id
        self.first_level_space = 2 if demo_id <= 2 else 1
        self.larger_res = self.first_level_space * res
        self.v_copies = 4 if self.demo_id <= 2 else 1
        self.dt = 0.03
        # self.dt = 0.01
        self.v_quant = v_quant
        self.dye_quant = dye_quant
        self.dye_type = dye_type

        self.mg_level = 0

        if self.dim == 2:
            mg_block_dim = 8
            target_block_dim = 128
        else:
            mg_block_dim = 4
            target_block_dim = 32
            # target_block_dim = 8
        self.block_dim = mg_block_dim
        # The MGPCG solver has an assumption that after mg_level levels of coarsening, the sparsity still follows block size of mg_block_dim

        while self.block_dim < target_block_dim:
            self.mg_level += 1
            self.block_dim *= 2

        assert self.block_dim == target_block_dim

        self.padding = self.block_dim
        # Offset the grid so that (0, 0) is the center
        # Simulation domain is [-0.5, 0.5)^3, and ideally we should never touch the bounds
        # self.offset = (-res // 2, ) * self.dim
        self.offset = (-int(self.larger_res // 2), ) * self.dim
        assert res % self.block_dim == 0

        self.debug = False
        self.inv_dx = self.larger_res
        self.dx = 1 / self.inv_dx

        field_args = {
            'inv_dx': self.inv_dx,
            'res': (self.larger_res, ) * dim,
            'offset': self.offset
        }

        v_type = ti.f32
        dye_type = ti.f32

        if self.v_quant:
            v_type = ti.types.quant.fixed(frac=21, signed=True, range=2**5)
            if self.benchmark_id == 1:
                v_type = ti.types.quant.float(exp=5, frac=9)
            elif self.benchmark_id == 2:
                v_type = ti.types.quant.fixed(frac=10)

        if self.dye_quant:
            if self.dye_type == DyeType.SHARED_EXP:
                dye_type = ti.types.quant.float(exp=5, frac=9, signed=False)
            elif self.dye_type == DyeType.FIXED_POINT:
                dye_type = ti.types.quant.fixed(frac=10, signed=False)
            elif self.dye_type == DyeType.NON_SHREAD_EXP:
                dye_type = ti.types.quant.float(exp=5, frac=5, signed=False)
            else:
                assert False, 'please set correct dye type !'

        self.v = SparseFieldCollection(
            num_copies=self.v_copies,
            dtype=v_type,
            vector_width=self.dim,
            **field_args)

        self.dye = SparseFieldCollection(
            num_copies=3, dtype=dye_type, vector_width=3, **field_args)

        self.pressure = SparseField(dtype=float, **field_args)

        # Here we let brightness and pressure share a single channel to save memory
        self.brightness = self.pressure

        def make_block_cell():
            indices = ti.ijk if self.dim == 3 else ti.ij
            # N // b2 // b1
            #  b2
            #  b1
            block = ti.root.pointer(indices,
                                    self.larger_res // 4 // self.block_dim)
            block1 = block.pointer(indices, 4)
            cell = block1.dense(indices, self.block_dim)
            return block, cell

        block, cell = make_block_cell()
        self.block = block
        self.cell = cell

        cell.place(self.pressure.field, offset=self.offset)

        for v in self.v:
            if self.v_quant and self.demo_id <= 2:
                cell.bit_struct(64).place(
                    v.field, offset=self.offset)
            elif self.v_quant and self.demo_id == 3:
                cell.bit_struct(32).place(
                    v.field, offset=self.offset, shared_exponent=self.benchmark_id==1)
            else:
                cell.place(v.field, offset=self.offset)

        for dye in self.dye:
            # Use different pointer/dense fields to hold the dyes. This
            # decouples from the velocity field, so the listgen can be
            # eliminated!
            _, cell = make_block_cell()
            if self.dye_quant:
                cell.bit_struct(32).place(
                    dye.field,
                    offset=self.offset,
                    shared_exponent=self.dye_type == DyeType.SHARED_EXP)
            else:
                cell.place(dye.field, offset=self.offset)

        self.advection_op = AdvectionOp(rk_order=rk_order, advect_op=advect_op)
        self.dye_sum = ti.field(dtype=ti.f64, shape=())


@ti.data_oriented
class FluidSolver(FluidSolverBase):
    def __init__(self,
                 res,
                 rk_order,
                 advect_op,
                 reflection,
                 dim,
                 source_radius=0.01,
                 demo_id=0,
                 v_quant=False,
                 dye_quant=False,
                 dye_type=0):
        super().__init__(
            res=res,
            rk_order=rk_order,
            advect_op=advect_op,
            dim=dim,
            demo_id=demo_id,
            v_quant=v_quant,
            dye_quant=dye_quant,
            dye_type=dye_type)

        self.dye_decay = 0.99
        self.reflection = reflection
        self.source_radius = source_radius
        self.dump_dye_rgb = True
        self.smoke_type = SmokeType.INIT_BY_SHAPE if self.demo_id == 0 else SmokeType.FLOW

        if reflection:
            self.dt *= 0.5
        self.debug = False
        self.T = 0

        self.density_map_block_split = 2
        assert self.block_dim % self.density_map_block_split == 0

        self.density_map_voxel_coverage = self.block_dim // self.density_map_block_split
        density_map_res = (
            self.larger_res // self.density_map_voxel_coverage, ) * self.dim
        self.coarsened_density_map = ti.field(
            dtype=ti.i32, shape=density_map_res)

        self.average_v = ti.Vector.field(self.dim, dtype=ti.f32, shape=())

        self.mgpcg = MGPCG(
            dim=self.dim,
            N=self.larger_res,
            offset=self.offset,
            n_mg_levels=self.mg_level,
            use_multigrid=True)

        
        if self.demo_id == 1:
            # flow injected from left bottom
            self.source_positions = [self.block_dim // 2]* self.dim
            self.source_positions[0] = self.block_dim // 2 - int(0.25 * self.larger_res)
            self.source_positions[1] = self.block_dim // 2 - int(0.2 * self.larger_res)
            self.populate_source()
        elif self.demo_id == 2:
            # flow appears from the bottom of the screen
            self.source_positions = [0.] * self.dim
            self.source_positions[1] = self.block_dim // 2 - int(
                0.25 * self.larger_res)
            self.populate_source()
        elif self.demo_id == 0:
            # smoke initialized with shape of bunny
            self.load_custom_source('inputs/640.np', res=640)
            self.set_custom_source(res=640)

    @ti.kernel
    def populate_source(self):
        # ti.activate(self.block, [
        #     self.source_position[i] - self.offset[i] for i in range(self.dim)
        # ])
        I = ti.Vector([
            0,
        ] * self.dim)
        for i in ti.static(range(self.dim)):
            I[i] = ti.cast(self.source_positions[i], int)
        self.pressure.field[I] += 0.

    @ti.kernel
    def expand_domain(self, dye: ti.template()):
        for I in ti.grouped(self.coarsened_density_map):
            self.coarsened_density_map[I] = 0

        # Re-compute a density spatial histogram
        for I in ti.grouped(dye):
            density = dye[I].norm()
            if density > 0.1:
                # populate neighbors
                self.coarsened_density_map[(
                    I - ti.Vector(self.offset)
                ) // self.density_map_voxel_coverage] += 1

        # Expand the simulation domain
        for I in ti.grouped(self.coarsened_density_map):
            if self.coarsened_density_map[I] > 0:
                for J in ti.grouped(ti.ndrange(*[(-1, 2)] * self.dim)):
                    K = (I + J) * self.density_map_voxel_coverage
                    # activate sparse grid at `K + offset`
                    self.pressure.field[K + ti.Vector(self.offset)] += 0

    @ti.kernel
    def inflow(self, v: ti.template(), dyef: ti.template(), t: ti.f32,
               dt: ti.f32):
        for I in ti.grouped(v):
            d2 = 0.0
            for k in ti.static(range(self.dim)):
                d2 += (I[k] + 0.5 - self.source_positions[k])**2

            # Set velocity
            target_v = ti.Vector.zero(ti.f32, self.dim)
            if ti.static(self.demo_id == 1):
                target_v[0] = 5
            else:
                target_v[0] = ti.sin(t * 3) * 0.9
                target_v[1] = 4.0

            weight = ti.exp(-d2 * (4 / (self.res * self.source_radius)**2))
            v_weight = ti.exp(-weight * dt * 10)
            v[I] = v[I] * v_weight + (1 - v_weight) * target_v

            # Set dye color
            dc = dyef[I]
            color = ti.sin(
                ti.Vector([t, t + math.pi * 1 / 3, t + math.pi * 2 / 3]) * 2)
            dc = dc + weight * (0.5 + color * 0.5) * 2

            if ti.static(self.demo_id == 1):
                dc_norm = dc.norm()
                dc_norm = min(dc_norm, 1.)
                v[I] = v[I] + dc_norm * ti.Vector([0.0, 3., 0.0]) * dt

            dc = dc * self.dye_decay
            dyef[I] = dc

    
    @ti.kernel
    def buoyancy(self, v: ti.template(), dyef: ti.template(), t: ti.f32,
                 dt: ti.f32):

        for I in ti.grouped(v):
            dc = dyef[I]
            dc_norm = dc.norm()
            dc_norm = min(dc_norm, 1.)
            v[I] = v[I] + dc_norm * ti.Vector([0.0, 3., 0.0]) * dt
            dyef[I] = dc * self.dye_decay

    @ti.kernel
    def compute_div_and_init_pressure_solver(self, v: ti.template()):
        for I in ti.grouped(v.field):
            div = 0.0
            for k in ti.static(range(self.dim)):
                ni = I  # neighbor index
                ni[k] -= 1
                vl = v.bounded_load(ni)[k]
                ni[k] += 2
                vr = v.bounded_load(ni)[k]
                div += vr - vl
            div *= 0.5
            self.mgpcg.init_r(I, -div)

    @ti.kernel
    def apply_pressure_with_adjustments(self, v: ti.template(),
                                        pressure: ti.template()):
        for I in ti.grouped(v.field):
            pdiff = ti.Vector.zero(ti.f32, self.dim)
            for k in ti.static(range(self.dim)):
                ni = I  # neighbor index
                ni[k] -= 1
                pl = pressure.bounded_load(ni)
                ni[k] += 2
                pr = pressure.bounded_load(ni)
                pdiff[k] = pr - pl
            v.field[I] = v.field[I] - 0.5 * pdiff + self.average_v[None]

    def advect(self, v, dt):
        # Use v as velocity field to advect self.v[0] into self.v[1], self.v[2] is tmp
        self.advection_op.advect(v, self.v[0], self.v[1], self.v[2], dt)
        self.advection_op.advect(v, self.dye[0], self.dye[1], self.dye[2], dt)

        self.dye.swap(0, 1)
        self.v.swap(0, 1)

    @ti.kernel
    def compute_average_v(self, v: ti.template()):
        tot = 0
        for I in ti.grouped(v):
            tot += 1
            self.average_v[None] += v[I]

        self.average_v[None] = (1 / tot) * self.average_v[None]

    def project(self, v, enforce_zero_average=False):

        self.mgpcg.reset()
        self.compute_div_and_init_pressure_solver(v)
        self.mgpcg.solve(max_iters=12, verbose=True)
        self.mgpcg.fetch_result(self.pressure.field)
        self.average_v[None] = [0] * self.dim

        if enforce_zero_average:
            self.compute_average_v(v)

        self.apply_pressure_with_adjustments(v, self.pressure)

    @ti.kernel
    def copy(self, a: ti.template(), b: ti.template()):
        for I in ti.grouped(a):
            a[I] = b[I]

    @ti.kernel
    def reflect_field(self, a: ti.template(), b: ti.template()):
        for I in ti.grouped(a):
            a[I] = b[I] * 2 - a[I]

    def reflect(self):
        self.copy(self.v[3].field, self.v[0].field)
        self.project(self.v[3])
        # v = v_divfree * 2 - v
        self.reflect_field(self.v[0].field, self.v[3].field)

    def step(self):
        if self.reflection:
            self.advect(self.v[0], self.dt)
            if self.smoke_type == SmokeType.FLOW:
                self.inflow(self.v[0].field, self.dye[0].field, self.T, self.dt)
            elif self.smoke_type == SmokeType.INIT_BY_SHAPE:
                self.buoyancy(self.v[0].field, self.dye[0].field, self.T, self.dt)
            else:
                assert False, 'please set correct smoke type'
            self.reflect()
            self.advect(self.v[3], self.dt)
            self.project(self.v[0])
            self.T += self.dt * 2
        else:
            self.advect(self.v[0], self.dt)
            self.inflow(self.v[0].field, self.dye[0].field, self.T, self.dt)
            self.project(self.v[0])
            self.T += self.dt
        self.expand_domain(self.dye[0].field)

    @ti.kernel
    def fetch_color_helper(self, v: ti.template(), arr: ti.types.ndarray()):
        if ti.static(self.dim == 3):
            for i, j, k in ti.ndrange(*((-self.res // 2,
                                         self.res // 2), ) * self.dim):
                I = ti.Vector([i, j, k])
                J = I + ti.Vector(
                    [self.res // 2, self.res // 2, self.res // 2])
                for p in ti.static(range(v.n)):
                    arr[J, p] = v[I][p]
        else:
            for i, j in ti.ndrange(*((-self.res // 2,
                                      self.res // 2), ) * self.dim):
                I = ti.Vector([i, j])
                J = I + ti.Vector([self.res // 2, self.res // 2])
                for p in ti.static(range(v.n)):
                    arr[J, p] = v[I][p]

    def fetch_color(self):
        arr = np.zeros((self.res, ) * self.dim + (3, ), dtype=np.float32)
        self.fetch_color_helper(self.dye[0].field, arr)
        return arr

    @ti.kernel
    def fetch_color_by_slice_helper(self, v: ti.template(), arr: ti.types.ndarray()):
        if ti.static(self.dim == 3):
            for i, j in ti.ndrange(*((-self.res // 2, self.res // 2), ) * 2):
                z = self.block_dim // 2
                if ti.static(self.demo_id == 0):
                    z = -200
                I = ti.Vector([i, j, z])
                J = ti.Vector([i + self.res // 2, j + self.res // 2])
                for p in ti.static(range(v.n)):
                    arr[J, p] = v[I][p]
        else:
            for i, j in ti.ndrange(*((-self.res // 2, self.res // 2), ) * 2):
                I = ti.Vector([i, j])
                J = ti.Vector([i + self.res // 2, j + self.res // 2])
                for p in ti.static(range(v.n)):
                    arr[J, p] = 1.0 - v[I][p]

    def fetch_color_by_slice(self):
        arr = np.zeros((self.res, ) * 2 + (3, ), dtype=np.float32)
        self.fetch_color_by_slice_helper(self.dye[0].field, arr)
        return arr

    @ti.kernel
    def fetch_vector_to_numpy_helper(self, v: ti.template(),
                                     arr: ti.types.ndarray()):
        for I in ti.grouped(v):
            for p in ti.static(range(v.n)):
                arr[I - ti.Vector(self.offset), p] = v[I][p]

    # TODO: should to_numpy_with_offset be part of Taichi?
    def fetch_vector_to_numpy(self, v):
        ret = np.zeros(v.shape + (v.field.n, ), dtype=np.float32)
        self.fetch_vector_to_numpy_helper(v.field, ret)
        return ret

    @ti.kernel
    def fetch_grid_activation_helper(self, v: ti.template(),
                                     arr: ti.types.ndarray()):
        if ti.static(self.dim == 3):
            for i, j, k in ti.ndrange(*((-self.res // 2,
                                         self.res // 2), ) * self.dim):
                I = ti.Vector([i, j, k])
                J = I + ti.Vector(
                    [self.res // 2, self.res // 2, self.res // 2])
                K = I - ti.Vector(self.offset)
                arr[J] = ti.is_active(self.block, K)
        else:
            for i, j in ti.ndrange(*((-self.res // 2,
                                      self.res // 2), ) * self.dim):
                I = ti.Vector([i, j])
                J = I + ti.Vector([self.res // 2, self.res // 2])
                K = I - ti.Vector(self.offset)
                arr[J] = ti.is_active(self.block, K)

    def fetch_grid_activation(self):
        ret = np.zeros(
            [
                self.res,
            ] * self.dim, dtype=np.float32)
        self.fetch_grid_activation_helper(self.v[0].field, ret)
        return ret

    @ti.kernel
    def fetch_grid_activation_by_slice_helper(self, v: ti.template(),
                                              arr: ti.types.ndarray()):
        # for I in ti.grouped(v):
        #     J = I - ti.Vector(self.offset)
        #     arr[J] = ti.is_active(self.block, J)
        res = self.res
        if ti.static(self.dim == 3):
            for i, j in ti.ndrange(*((-res // 2, res // 2), ) * 2):
                I = ti.Vector([i, j, self.block_dim // 2])
                J = ti.Vector([i + res // 2, j + res // 2])
                K = I - ti.Vector(self.offset)
                arr[J] = ti.is_active(self.block, K)
        else:
            for i, j in ti.ndrange(*((-self.res // 2,
                                      self.res // 2), ) * self.dim):
                I = ti.Vector([i, j])
                J = I + ti.Vector([self.res // 2, self.res // 2])
                K = I - ti.Vector(self.offset)
                arr[J] = ti.is_active(self.block, K)

    def fetch_grid_activation_by_slice(self):
        res = self.res
        ret = np.zeros(
            [
                res,
            ] * 2, dtype=np.float32)
        self.fetch_grid_activation_by_slice_helper(self.v[0].field, ret)
        return ret

    @ti.kernel
    def fetch_density(self, density_rgb: ti.template(),
                      density_np: ti.types.ndarray()):
        for I in ti.grouped(density_rgb):
            J = I - ti.Vector(self.offset)
            d = ti.cast(min(density_rgb[I].norm(), 1.0) * 100, ti.u8)
            density_np[J] = d

    @ti.kernel
    def fetch_density_by_slice(self, density_rgb: ti.template(),
                               density_np: ti.types.ndarray(), k: ti.int32):
        for i, j in ti.ndrange(*((-self.res // 2, self.res // 2), ) * 2):
            I = ti.Vector([k, i, j])
            J = ti.Vector([i + self.res // 2, j + self.res // 2])
            if ti.static(self.dump_dye_rgb):
                for l in ti.static(range(3)):
                    d = ti.cast(density_rgb[I][l] * 100, ti.u8)
                    density_np[J, l] = d
            else:
                d = ti.cast(min(density_rgb[I].norm(), 1.0) * 100, ti.u8)
                density_np[J] = d

    def dump_density(self, fn):
        density = np.zeros((self.res, self.res, self.res, 3), dtype=np.uint8)
        for k in range(self.res):
            self.fetch_density_by_slice(self.dye[0].field, density[k],
                                        k - self.res // 2)
        density.tofile(fn)

    @ti.kernel
    def count_activated_voxels_helper(self, v: ti.template()) -> ti.i32:
        count = 0
        for I in ti.grouped(v):
            count += 1
        return count

    def count_activated_voxels(self):
        return self.count_activated_voxels_helper(self.v[0].field)

    def load_custom_source(self, fn, res):
        density_source = np.fromfile(fn, dtype=np.uint8)
        self.density_source = density_source.reshape(res, res, res)

    @ti.kernel
    def set_custom_source_helper(self, dyef: ti.template(), arr: ti.types.ndarray(),
                                 k: ti.int32):
        arr_res = arr.shape
        for i, j in ti.ndrange(*arr_res):
            I = ti.Vector([i, j])
            if arr[I] > 0:
                dyef[k - self.res // 2, i - self.res // 2,
                     j - self.res // 2] = [0., 0.5, 1.0]

    def set_custom_source(self, res):
        for k in range(res):
            self.set_custom_source_helper(self.dye[0].field, self.density_source[k],
                                          k)

    @ti.kernel
    def sum_density(self, dyef: ti.template()) -> ti.f64:
        self.dye_sum[None] = 0.0
        for I in ti.grouped(dyef):
            for k in ti.static(range(self.dim)):
                self.dye_sum[None] += dyef[I][k]
        return self.dye_sum[None]
