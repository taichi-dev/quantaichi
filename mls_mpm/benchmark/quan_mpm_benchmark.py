import taichi as ti
import numpy as np
import time
import numbers
import math
import argparse
import struct
import multiprocessing as mp

USE_IN_BLENDER = False

ti.require_version(0, 7, 10)


@ti.func
def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


@ti.func
def inside_ccw(p, a, b, c):
    return cross2d(a - p, b - p) >= 0 and cross2d(
        b - p, c - p) >= 0 and cross2d(c - p, a - p) >= 0


@ti.data_oriented
class Voxelizer:
    def __init__(self, res, dx, super_sample=2, precision=ti.f64, padding=3):
        assert len(res) == 3
        # Super sample by 2x
        self.res = (res[0] * super_sample, res[1] * super_sample,
                    res[2] * super_sample)
        self.dx = dx / super_sample
        self.inv_dx = 1 / self.dx
        self.voxels = ti.field(ti.i32)
        self.block = ti.root.pointer(
            ti.ijk, (self.res[0] // 8, self.res[1] // 8, self.res[2] // 8))
        self.block.dense(ti.ijk, 8).place(self.voxels)

        assert precision in [ti.f32, ti.f64]
        self.precision = precision
        self.padding = padding

    @ti.func
    def fill(self, p, q, height, inc):
        for i in range(self.padding, height):
            self.voxels[p, q, i] += inc

    @ti.kernel
    def voxelize_triangles(self, num_triangles: ti.i32,
                           triangles: ti.ext_arr()):
        for i in range(num_triangles):
            jitter_scale = ti.cast(0, self.precision)
            if ti.static(self.precision is ti.f32):
                jitter_scale = 1e-4
            else:
                jitter_scale = 1e-8
            # We jitter the vertices to prevent voxel samples from lying precicely at triangle edges
            jitter = ti.Vector([
                -0.057616723909439505, -0.25608986292614977,
                0.06716309129743714
            ]) * jitter_scale
            a = ti.Vector([triangles[i, 0], triangles[i, 1], triangles[i, 2]
                           ]) + jitter
            b = ti.Vector([triangles[i, 3], triangles[i, 4], triangles[i, 5]
                           ]) + jitter
            c = ti.Vector([triangles[i, 6], triangles[i, 7], triangles[i, 8]
                           ]) + jitter

            bound_min = ti.Vector.zero(self.precision, 3)
            bound_max = ti.Vector.zero(self.precision, 3)
            for k in ti.static(range(3)):
                bound_min[k] = min(a[k], b[k], c[k])
                bound_max[k] = max(a[k], b[k], c[k])

            p_min = int(ti.floor(bound_min[0] * self.inv_dx))
            p_max = int(ti.floor(bound_max[0] * self.inv_dx)) + 1

            p_min = max(self.padding, p_min)
            p_max = min(self.res[0] - self.padding, p_max)

            q_min = int(ti.floor(bound_min[1] * self.inv_dx))
            q_max = int(ti.floor(bound_max[1] * self.inv_dx)) + 1

            q_min = max(self.padding, q_min)
            q_max = min(self.res[1] - self.padding, q_max)

            normal = ((b - a).cross(c - a)).normalized()

            if abs(normal[2]) < 1e-10:
                continue

            a_proj = ti.Vector([a[0], a[1]])
            b_proj = ti.Vector([b[0], b[1]])
            c_proj = ti.Vector([c[0], c[1]])

            for p in range(p_min, p_max):
                for q in range(q_min, q_max):
                    pos2d = ti.Vector([(p + 0.5) * self.dx,
                                       (q + 0.5) * self.dx])
                    if inside_ccw(pos2d, a_proj, b_proj, c_proj) or inside_ccw(
                            pos2d, a_proj, c_proj, b_proj):
                        base_voxel = ti.Vector([pos2d[0], pos2d[1], 0])
                        height = int(-normal.dot(base_voxel - a) / normal[2] *
                                     self.inv_dx + 0.5)
                        height = min(height, self.res[1] - self.padding)
                        inc = 0
                        if normal[2] > 0:
                            inc = 1
                        else:
                            inc = -1
                        self.fill(p, q, height, inc)

    def voxelize(self, triangles):
        assert isinstance(triangles, np.ndarray)
        triangles = triangles.astype(np.float64)
        assert triangles.dtype in [np.float32, np.float64]
        if self.precision is ti.f32:
            triangles = triangles.astype(np.float32)
        elif self.precision is ti.f64:
            triangles = triangles.astype(np.float64)
        else:
            assert False
        assert len(triangles.shape) == 2
        assert triangles.shape[1] == 9

        self.block.deactivate_all()
        num_triangles = len(triangles)
        self.voxelize_triangles(num_triangles, triangles)


@ti.data_oriented
class MPMSolver:
    material_water = 0
    material_elastic = 1
    material_snow = 2
    material_sand = 3
    materials = {
        'WATER': material_water,
        'ELASTIC': material_elastic,
        'SNOW': material_snow,
        'SAND': material_sand
    }

    # Surface boundary conditions

    # Stick to the boundary
    surface_sticky = 0
    # Slippy boundary
    surface_slip = 1
    # Slippy and free to separate
    surface_separate = 2

    surfaces = {
        'STICKY': surface_sticky,
        'SLIP': surface_slip,
        'SEPARATE': surface_separate
    }

    grid_size = 4096

    def __init__(
            self,
            res,
            quant=False,
            size=1,
            max_num_particles=2 ** 30,
            # Max 1 G particles
            padding=3,
            unbounded=False,
            dt_scale=1.0,
            E_scale=1.0,
            vol_scale=1.0,
            nu=0.2,
            voxelizer_super_sample=2,
            use_g2p2g=True,
            use_bls=True,
            g2p2g_allowed_cfl=0.9,  # 0.0 for no CFL limit
            water_density=1.0,
            support_plasticity=True,
            use_gpu=True):
        self.dim = len(res)
        self.quant = quant
        self.use_g2p2g = use_g2p2g
        self.use_bls = use_bls
        self.g2p2g_allowed_cfl = g2p2g_allowed_cfl
        self.water_density = water_density
        assert self.dim in (
            2, 3), "MPM solver supports only 2D and 3D simulations."

        self.use_gpu = use_gpu
        self.size = size
        self.total_time = 0.0
        self.total_kernel_time = 0.0

        self.t = 0.0
        self.res = res
        self.n_particles = ti.field(ti.i32, shape=())
        self.dx = size / res[0]
        self.inv_dx = 1.0 / self.dx
        self.default_dt = 2e-2 * self.dx / size * dt_scale
        self.p_vol = self.dx ** self.dim * vol_scale
        self.p_rho = 1000
        self.p_mass = self.p_vol * self.p_rho
        self.max_num_particles = max_num_particles
        self.gravity = ti.Vector.field(self.dim, dtype=ti.f32, shape=())
        self.source_bound = ti.Vector.field(self.dim, dtype=ti.f32, shape=2)
        self.source_velocity = ti.Vector.field(self.dim,
                                               dtype=ti.f32,
                                               shape=())
        self.input_grid = 0
        self.all_time_max_velocity = 0.0
        self.support_plasticity = support_plasticity
        self.F_bound = 4.0

        # affine velocity field
        if not self.use_g2p2g:
            self.C = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32)
        # deformation gradient

        if quant:
            assert self.dim == 3
            ci21 = ti.type_factory.custom_int(21, True)
            cft = ti.type_factory.custom_float(significand_type=ci21,
                                               scale=1 / (2 ** 19))
            self.x = ti.Vector.field(self.dim, dtype=cft)

            cu6 = ti.type_factory.custom_int(7, False)
            ci19 = ti.type_factory.custom_int(19, True)
            cft = ti.type_factory.custom_float(significand_type=ci19,
                                               exponent_type=cu6)
            self.v = ti.Vector.field(self.dim, dtype=cft)

            ci16 = ti.type_factory.custom_int(16, True)
            cft = ti.type_factory.custom_float(significand_type=ci16,
                                               scale=(self.F_bound + 0.1) /
                                                     (2 ** 15))
            self.F = ti.Matrix.field(self.dim, self.dim, dtype=cft)
        else:
            self.v = ti.Vector.field(self.dim, dtype=ti.f32)
            self.x = ti.Vector.field(self.dim, dtype=ti.f32)
            self.F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32)

        self.last_time_final_particles = ti.field(dtype=ti.i32, shape=())
        # material id
        if quant:
            self.material = ti.field(dtype=ti.quant.int(16, False))
            self.color = ti.field(dtype=ti.i32)
        else:
            self.material = ti.field(dtype=ti.i32)
            self.color = ti.field(dtype=ti.i32)
        # plastic deformation volume ratio
        if self.support_plasticity:
            self.Jp = ti.field(dtype=ti.f32)

        if self.dim == 2:
            indices = ti.ij
        else:
            indices = ti.ijk

        offset = tuple(-self.grid_size // 2 for _ in range(self.dim))
        self.offset = offset

        self.num_grids = 2 if self.use_g2p2g else 1

        grid_block_size = 128
        if self.dim == 2:
            self.leaf_block_size = 16
        else:
            # TODO: use 8?
            self.leaf_block_size = 4

        self.grid = []
        self.grid_v = []
        self.grid_m = []
        self.pid = []

        for g in range(self.num_grids):
            # grid node momentum/velocity
            grid_v = ti.Vector.field(self.dim, dtype=ti.f32)
            grid_m = ti.field(dtype=ti.f32)
            pid = ti.field(ti.i32)
            self.grid_v.append(grid_v)
            # grid node mass
            self.grid_m.append(grid_m)
            grid = ti.root.pointer(indices, self.grid_size // grid_block_size)
            block = grid.pointer(indices,
                                 grid_block_size // self.leaf_block_size)
            self.grid.append(grid)

            def block_component(c):
                block.dense(indices, self.leaf_block_size).place(c,
                                                                 offset=offset)

            block_component(grid_m)
            for v in grid_v.entries:
                block_component(v)

            self.pid.append(pid)
            block.dynamic(ti.indices(self.dim),
                          1024 * 1024,
                          chunk_size=self.leaf_block_size ** self.dim * 8).place(
                pid, offset=offset + (0,))

        self.padding = padding

        # Young's modulus and Poisson's ratio
        self.E, self.nu = 1e6 * size * E_scale, nu
        # Lame parameters
        self.mu_0, self.lambda_0 = self.E / (
                2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) *
                                                        (1 - 2 * self.nu))

        # Sand parameters
        friction_angle = math.radians(45)
        sin_phi = math.sin(friction_angle)
        self.alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

        self.particle = ti.root.dynamic(ti.i, max_num_particles, 2 ** 23)

        if self.quant:
            if not self.use_g2p2g:
                self.particle.place(self.C)
            if self.support_plasticity:
                self.particle.place(self.Jp)
            self.particle.bit_struct(num_bits=64).place(self.x)
            self.particle.bit_struct(num_bits=64).place(self.v,
                                                        shared_exponent=True)
            self.particle.bit_struct(num_bits=32).place(
                self.F(0, 0), self.F(0, 1))
            self.particle.bit_struct(num_bits=32).place(
                self.F(0, 2), self.F(1, 0))
            self.particle.bit_struct(num_bits=32).place(
                self.F(1, 1), self.F(1, 2))
            self.particle.bit_struct(num_bits=32).place(
                self.F(2, 0), self.F(2, 1))
            self.particle.bit_struct(num_bits=32).place(
                self.F(2, 2), self.material)
            self.particle.place(self.color)
        else:
            self.particle.place(self.x, self.v, self.F, self.material,
                                self.color)
            if self.support_plasticity:
                self.particle.place(self.Jp)
            if not self.use_g2p2g:
                self.particle.place(self.C)

        self.total_substeps = 0
        self.unbounded = unbounded

        if self.dim == 2:
            self.voxelizer = None
            self.set_gravity((0, -9.8))
        else:
            self.voxelizer = Voxelizer(res=self.res,
                                       dx=self.dx,
                                       padding=self.padding,
                                       super_sample=voxelizer_super_sample)
            self.set_gravity((0, -9.8, 0))

        self.voxelizer_super_sample = voxelizer_super_sample

        self.grid_postprocess = []

        self.add_bounding_box(self.unbounded)

        self.writers = []

        if not self.use_g2p2g:
            self.grid = self.grid[0]
            self.grid_v = self.grid_v[0]
            self.grid_m = self.grid_m[0]
            self.pid = self.pid[0]

    def stencil_range(self):
        return ti.ndrange(*((3,) * self.dim))

    def set_gravity(self, g):
        assert isinstance(g, (tuple, list))
        assert len(g) == self.dim
        self.gravity[None] = g

    @ti.func
    def sand_projection(self, sigma, p):
        sigma_out = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        epsilon = ti.Vector.zero(ti.f32, self.dim)
        for i in ti.static(range(self.dim)):
            epsilon[i] = ti.log(max(abs(sigma[i, i]), 1e-4))
            sigma_out[i, i] = 1
        tr = epsilon.sum() + self.Jp[p]
        epsilon_hat = epsilon - tr / self.dim
        epsilon_hat_norm = epsilon_hat.norm() + 1e-20
        if tr >= 0.0:
            self.Jp[p] = tr
        else:
            self.Jp[p] = 0.0
            delta_gamma = epsilon_hat_norm + (
                    self.dim * self.lambda_0 +
                    2 * self.mu_0) / (2 * self.mu_0) * tr * self.alpha
            for i in ti.static(range(self.dim)):
                sigma_out[i, i] = ti.exp(epsilon[i] - max(0, delta_gamma) /
                                         epsilon_hat_norm * epsilon_hat[i])

        return sigma_out

    @ti.kernel
    def build_pid(self, pid: ti.template(), offset: ti.template()):
        ti.block_dim(64)
        for p in self.x:
            base = int(ti.floor(self.x[p] * self.inv_dx - offset))
            ti.append(pid.parent(), base - ti.Vector(list(self.offset)), p)

    @ti.kernel
    def g2p2g(self, dt: ti.f32, pid: ti.template(), grid_v_in: ti.template(),
              grid_v_out: ti.template(), grid_m_out: ti.template()):
        ti.block_dim(512)
        ti.no_activate(self.particle)
        if ti.static(self.use_bls):
            ti.block_local(*grid_v_in.entries)
            ti.block_local(grid_m_out)
            ti.block_local(*grid_v_out.entries)
        for I in ti.grouped(pid):
            p = pid[I]
            # G2P
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2
            ]
            new_v = ti.Vector.zero(ti.f32, self.dim)
            C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            # loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(float) - fx
                g_v = grid_v_in[base + offset]
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
                C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            if p >= self.last_time_final_particles[None]:
                # New particles. No G2P.
                new_v = self.v[p]
                C = ti.Matrix.zero(ti.f32, self.dim, self.dim)

            if ti.static(self.g2p2g_allowed_cfl > 0):
                v_allowed = self.dx * self.g2p2g_allowed_cfl / dt
                for d in ti.static(range(self.dim)):
                    new_v[d] = min(max(new_v[d], -v_allowed), v_allowed)

            self.v[p] = new_v
            self.x[p] += dt * self.v[p]  # advection

            # P2G
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], -1, 2)

            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w2 = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2,
                  0.5 * (fx - 0.5) ** 2]
            # deformation gradient update
            new_F = (ti.Matrix.identity(ti.f32, self.dim) + dt * C) @ self.F[p]
            if ti.static(self.quant):
                new_F = max(-self.F_bound, min(self.F_bound, new_F))
            self.F[p] = new_F
            # Hardening coefficient: snow gets harder when compressed
            h = 1.0
            if ti.static(self.support_plasticity):
                h = ti.exp(10 * (1.0 - self.Jp[p]))
            if self.material[
                p] == self.material_elastic:  # jelly, make it softer
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == self.material_water:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            if self.material[p] != self.material_sand:
                for d in ti.static(range(self.dim)):
                    new_sig = sig[d, d]
                    if self.material[p] == self.material_snow:  # Snow
                        new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                      1 + 4.5e-3)  # Plasticity
                    if ti.static(self.support_plasticity):
                        self.Jp[p] *= sig[d, d] / new_sig
                    sig[d, d] = new_sig
                    J *= new_sig
            if self.material[p] == self.material_water:
                # Reset deformation gradient to avoid numerical instability
                new_F = ti.Matrix.identity(ti.f32, self.dim)
                new_F[0, 0] = J
                self.F[p] = new_F
            elif self.material[p] == self.material_snow:
                # Reconstruct elastic deformation gradient after plasticity
                self.F[p] = U @ sig @ V.transpose()

            stress = ti.Matrix.zero(ti.f32, self.dim, self.dim)

            if self.material[p] != self.material_sand:
                stress = 2 * mu * (
                        self.F[p] - U @ V.transpose()) @ self.F[p].transpose(
                ) + ti.Matrix.identity(ti.f32, self.dim) * la * J * (J - 1)
            else:
                if ti.static(self.support_plasticity):
                    sig = self.sand_projection(sig, p)
                    self.F[p] = U @ sig @ V.transpose()
                    log_sig_sum = 0.0
                    center = ti.Matrix.zero(ti.f32, self.dim, self.dim)
                    for i in ti.static(range(self.dim)):
                        log_sig_sum += ti.log(sig[i, i])
                        center[i, i] = 2.0 * self.mu_0 * ti.log(
                            sig[i, i]) * (1 / sig[i, i])
                    for i in ti.static(range(self.dim)):
                        center[i,
                               i] += self.lambda_0 * log_sig_sum * (1 /
                                                                    sig[i, i])
                    stress = U @ center @ V.transpose() @ self.F[p].transpose()

            stress = (-dt * self.p_vol * 4 * self.inv_dx ** 2) * stress
            affine = stress + self.p_mass * C

            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w2[offset[d]][d]
                grid_v_out[base +
                           offset] += weight * (self.p_mass * self.v[p] +
                                                affine @ dpos)
                grid_m_out[base + offset] += weight * self.p_mass

        self.last_time_final_particles[None] = self.n_particles[None]

    @ti.kernel
    def p2g(self, dt: ti.f32):
        ti.no_activate(self.particle)
        ti.block_dim(256)
        if ti.static(self.use_bls):
            ti.block_local(*self.grid_v.entries)
            ti.block_local(self.grid_m)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2,
                 0.5 * (fx - 0.5) ** 2]
            # deformation gradient update
            F = self.F[p]
            if self.material[p] == self.material_water:  # liquid
                F = ti.Matrix.identity(ti.f32, self.dim)
                F[0, 0] = self.Jp[p]

            F = (ti.Matrix.identity(ti.f32, self.dim) +
                 dt * self.C[p]) @ F
            # Hardening coefficient: snow gets harder when compressed
            h = 1.0
            if ti.static(self.support_plasticity):
                if self.material[p] != self.material_water:
                    h = ti.exp(10 * (1.0 - self.Jp[p]))
            if self.material[
                p] == self.material_elastic:  # jelly, make it softer
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == self.material_water:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(F)
            J = 1.0
            if self.material[p] != self.material_sand:
                for d in ti.static(range(self.dim)):
                    new_sig = sig[d, d]
                    if self.material[p] == self.material_snow:  # Snow
                        new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                      1 + 4.5e-3)  # Plasticity
                    if ti.static(self.support_plasticity):
                        self.Jp[p] *= sig[d, d] / new_sig
                    sig[d, d] = new_sig
                    J *= new_sig
            if self.material[p] == self.material_water:
                # Reset deformation gradient to avoid numerical instability
                F = ti.Matrix.identity(ti.f32, self.dim)
                F[0, 0] = J
                if ti.static(self.support_plasticity):
                    self.Jp[p] = J
            elif self.material[p] == self.material_snow:
                # Reconstruct elastic deformation gradient after plasticity
                F = U @ sig @ V.transpose()

            stress = ti.Matrix.zero(ti.f32, self.dim, self.dim)

            if self.material[p] != self.material_sand:
                stress = 2 * mu * (F - U @ V.transpose()) @ F.transpose(
                ) + ti.Matrix.identity(ti.f32, self.dim) * la * J * (J - 1)
            else:
                if ti.static(self.support_plasticity):
                    sig = self.sand_projection(sig, p)
                    F = U @ sig @ V.transpose()
                    log_sig_sum = 0.0
                    center = ti.Matrix.zero(ti.f32, self.dim, self.dim)
                    for i in ti.static(range(self.dim)):
                        log_sig_sum += ti.log(sig[i, i])
                        center[i, i] = 2.0 * self.mu_0 * ti.log(
                            sig[i, i]) * (1 / sig[i, i])
                    for i in ti.static(range(self.dim)):
                        center[i,
                               i] += self.lambda_0 * log_sig_sum * (1 /
                                                                    sig[i, i])
                    stress = U @ center @ V.transpose() @ F.transpose()
            self.F[p] = F

            stress = (-dt * self.p_vol * 4 * self.inv_dx ** 2) * stress
            # TODO: implement g2p2g pmass
            mass = self.p_mass
            if self.material[p] == self.material_water:
                mass *= self.water_density
            affine = stress + mass * self.C[p]

            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_v[base + offset] += weight * (mass * self.v[p] +
                                                        affine @ dpos)
                self.grid_m[base + offset] += weight * mass

    @ti.kernel
    def grid_normalization_and_gravity(self, dt: ti.f32, grid_v: ti.template(),
                                       grid_m: ti.template()):
        for I in ti.grouped(grid_m):
            if grid_m[I] > 0:  # No need for epsilon here
                grid_v[I] = (1 / grid_m[I]) * grid_v[I]  # Momentum to velocity
                grid_v[I] += dt * self.gravity[None]

    @ti.kernel
    def grid_bounding_box(self, t: ti.f32, dt: ti.f32,
                          unbounded: ti.template(), grid_v: ti.template()):
        for I in ti.grouped(grid_v):
            for d in ti.static(range(self.dim)):
                if ti.static(unbounded):
                    if I[d] < -self.grid_size // 2 + self.padding and grid_v[
                        I][d] < 0:
                        grid_v[I][d] = 0  # Boundary conditions
                    if I[d] >= self.grid_size // 2 - self.padding and grid_v[
                        I][d] > 0:
                        grid_v[I][d] = 0
                else:
                    if I[d] < self.padding and grid_v[I][d] < 0:
                        grid_v[I][d] = 0  # Boundary conditions
                    if I[d] >= self.res[d] - self.padding and grid_v[I][d] > 0:
                        grid_v[I][d] = 0

    def add_sphere_collider(self, center, radius, surface=surface_sticky):
        center = list(center)

        @ti.kernel
        def collide(t: ti.f32, dt: ti.f32):
            for I in ti.grouped(self.grid_m):
                offset = I * self.dx - ti.Vector(center)
                if offset.norm_sqr() < radius * radius:
                    if ti.static(surface == self.surface_sticky):
                        self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
                    else:
                        v = self.grid_v[I]
                        normal = offset.normalized(1e-5)
                        normal_component = normal.dot(v)

                        if ti.static(surface == self.surface_slip):
                            # Project out all normal component
                            v = v - normal * normal_component
                        else:
                            # Project out only inward normal component
                            v = v - normal * min(normal_component, 0)

                        self.grid_v[I] = v

        self.grid_postprocess.append(collide)

    def add_surface_collider(self,
                             point,
                             normal,
                             surface=surface_sticky,
                             friction=0.0):
        point = list(point)
        # normalize normal
        normal_scale = 1.0 / math.sqrt(sum(x ** 2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        if surface == self.surface_sticky and friction != 0:
            raise ValueError('friction must be 0 on sticky surfaces.')

        @ti.kernel
        def collide(t: ti.f32, dt: ti.f32, grid_v: ti.template()):
            for I in ti.grouped(grid_v):
                offset = I * self.dx - ti.Vector(point)
                n = ti.Vector(normal)
                if offset.dot(n) < 0:
                    if ti.static(surface == self.surface_sticky):
                        grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
                    else:
                        v = grid_v[I]
                        normal_component = n.dot(v)

                        if ti.static(surface == self.surface_slip):
                            # Project out all normal component
                            v = v - n * normal_component
                        else:
                            # Project out only inward normal component
                            v = v - n * min(normal_component, 0)

                        if normal_component < 0 and v.norm() > 1e-30:
                            # apply friction here
                            v = v.normalized() * max(
                                0,
                                v.norm() + normal_component * friction)

                        grid_v[I] = v

        self.grid_postprocess.append(collide)

    def add_bounding_box(self, unbounded):
        self.grid_postprocess.append(
            lambda t, dt, grid_v: self.grid_bounding_box(
                t, dt, unbounded, grid_v))

    @ti.kernel
    def g2p(self, dt: ti.f32):
        ti.block_dim(256)
        if ti.static(self.use_bls):
            ti.block_local(*self.grid_v.entries)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2
            ]
            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            # loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(float) - fx
                g_v = self.grid_v[base + offset]
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += dt * self.v[p]  # advection

    @ti.kernel
    def compute_max_velocity(self) -> ti.f32:
        max_velocity = 0.0
        for p in self.v:
            v = self.v[p]
            v_max = 0.0
            for i in ti.static(range(self.dim)):
                v_max = max(v_max, abs(v[i]))
            ti.atomic_max(max_velocity, v_max)
        return max_velocity

    def step(self, frame_dt, print_stat=False, add_to_stat=True):
        initial_kernel_t = ti.kernel_profiler_total_time()
        begin_t = time.time()
        begin_substep = self.total_substeps

        substeps = int(frame_dt / self.default_dt - 0.1) + 1

        if print_stat:
            print(f'needed substeps: {substeps}')
        for i in range(substeps):
            print('.', end='', flush=True)
            self.total_substeps += 1
            dt = frame_dt / substeps

            if self.use_g2p2g:
                output_grid = 1 - self.input_grid
                self.grid[output_grid].deactivate_all()
                self.build_pid(self.pid[self.input_grid], 0.5)
                self.g2p2g(dt, self.pid[self.input_grid],
                           self.grid_v[self.input_grid],
                           self.grid_v[output_grid], self.grid_m[output_grid])
                self.grid_normalization_and_gravity(dt,
                                                    self.grid_v[output_grid],
                                                    self.grid_m[output_grid])
                for p in self.grid_postprocess:
                    p(self.t, dt, self.grid_v[output_grid])
                self.input_grid = output_grid
                self.t += dt
            else:
                self.grid.deactivate_all()
                self.build_pid(self.pid, 0.5)
                self.p2g(dt)
                self.grid_normalization_and_gravity(dt, self.grid_v,
                                                    self.grid_m)
                for p in self.grid_postprocess:
                    p(self.t, dt, self.grid_v)
                self.t += dt
                self.g2p(dt)
        self.all_time_max_velocity = max(self.all_time_max_velocity,
                                         self.compute_max_velocity())
        print()
        frame_time = time.time() - begin_t
        frame_kernel_time = ti.kernel_profiler_total_time() - initial_kernel_t
        if add_to_stat:
            self.total_time += frame_time
            self.total_kernel_time += frame_kernel_time

        if print_stat:
            ti.kernel_profiler_print()
            try:
                ti.memory_profiler_print()
            except:
                pass
            print(f'CFL: {self.all_time_max_velocity * dt / self.dx}')
            print(f'num particles={self.n_particles[None]}')
            print(f'  frame time {time.time() - begin_t:.3f} s')
            print(
                f'  substep time {1000 * (time.time() - begin_t) / (self.total_substeps - begin_substep):.3f} ms'
            )

    @ti.func
    def seed_particle(self, i, x, material, color, velocity):
        self.x[i] = x
        self.v[i] = velocity
        self.F[i] = ti.Matrix.identity(ti.f32, self.dim)
        self.color[i] = color
        self.material[i] = material

        if ti.static(self.support_plasticity):
            if material == self.material_sand:
                self.Jp[i] = 0
            else:
                self.Jp[i] = 1

    @ti.kernel
    def seed(self, new_particles: ti.i32, new_material: ti.i32, color: ti.i32):
        for i in range(self.n_particles[None],
                       self.n_particles[None] + new_particles):
            self.material[i] = new_material
            x = ti.Vector.zero(ti.f32, self.dim)
            for k in ti.static(range(self.dim)):
                x[k] = self.source_bound[0][k] + ti.random(
                ) * self.source_bound[1][k]
            self.seed_particle(i, x, new_material, color,
                               self.source_velocity[None])

    def set_source_velocity(self, velocity):
        if velocity is not None:
            velocity = list(velocity)
            assert len(velocity) == self.dim
            self.source_velocity[None] = velocity
        else:
            for i in range(self.dim):
                self.source_velocity[None][i] = 0

    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 sample_density=None,
                 velocity=None):
        if sample_density is None:
            sample_density = 2 ** self.dim
        vol = 1
        for i in range(self.dim):
            vol = vol * cube_size[i]
        num_new_particles = int(sample_density * vol / self.dx ** self.dim + 1)
        assert self.n_particles[
                   None] + num_new_particles <= self.max_num_particles

        for i in range(self.dim):
            self.source_bound[0][i] = lower_corner[i]
            self.source_bound[1][i] = cube_size[i]

        self.set_source_velocity(velocity=velocity)

        self.seed(num_new_particles, material, color)
        self.n_particles[None] += num_new_particles

    @ti.kernel
    def add_texture_2d(self, offset_x: ti.f32, offset_y: ti.f32,
                       texture: ti.ext_arr()):
        for i, j in ti.ndrange(texture.shape[0], texture.shape[1]):
            if texture[i, j] > 0.1:
                pid = ti.atomic_add(self.n_particles[None], 1)
                x = ti.Vector([
                    offset_x + i * self.dx * 0.5, offset_y + j * self.dx * 0.5
                ])
                self.seed_particle(pid, x, self.material_elastic, 0xFFFFFF,
                                   self.source_velocity[None])

    @ti.func
    def random_point_in_unit_sphere(self):
        ret = ti.Vector.zero(ti.f32, n=self.dim)
        while True:
            for i in ti.static(range(self.dim)):
                ret[i] = ti.random(ti.f32) * 2 - 1
            if ret.norm_sqr() <= 1:
                break
        return ret

    @ti.kernel
    def seed_ellipsoid(self, new_particles: ti.i32, new_material: ti.i32,
                       color: ti.i32):

        for i in range(self.n_particles[None],
                       self.n_particles[None] + new_particles):
            x = self.source_bound[0] + self.random_point_in_unit_sphere(
            ) * self.source_bound[1]
            self.seed_particle(i, x, new_material, color,
                               self.source_velocity[None])

    def add_ellipsoid(self,
                      center,
                      radius,
                      material,
                      color=0xFFFFFF,
                      sample_density=None,
                      velocity=None):
        if sample_density is None:
            sample_density = 2 ** self.dim

        if isinstance(radius, numbers.Number):
            radius = [
                         radius,
                     ] * self.dim

        radius = list(radius)

        if self.dim == 2:
            num_particles = math.pi
        else:
            num_particles = 4 / 3 * math.pi

        for i in range(self.dim):
            num_particles *= radius[i] * self.inv_dx

        num_particles = int(math.ceil(num_particles * sample_density))

        self.source_bound[0] = center
        self.source_bound[1] = radius

        self.set_source_velocity(velocity=velocity)

        assert self.n_particles[None] + num_particles <= self.max_num_particles

        self.seed_ellipsoid(num_particles, material, color)
        self.n_particles[None] += num_particles

    @ti.kernel
    def seed_from_voxels(self, material: ti.i32, color: ti.i32,
                         sample_density: ti.i32):
        for i, j, k in self.voxelizer.voxels:
            inside = 1
            for d in ti.static(range(3)):
                inside = inside and -self.grid_size // 2 + self.padding <= i and i < self.grid_size // 2 - self.padding
            if inside and self.voxelizer.voxels[i, j, k] > 0:
                s = sample_density / self.voxelizer_super_sample ** self.dim
                for l in range(sample_density + 1):
                    if ti.random() + l < s:
                        x = ti.Vector([
                            ti.random() + i,
                            ti.random() + j,
                            ti.random() + k
                        ]) * (self.dx / self.voxelizer_super_sample
                              ) + self.source_bound[0]
                        p = ti.atomic_add(self.n_particles[None], 1)
                        self.seed_particle(p, x, material, color,
                                           self.source_velocity[None])

    def add_mesh(self,
                 triangles,
                 material,
                 color=0xFFFFFF,
                 sample_density=None,
                 velocity=None,
                 translation=None):
        assert self.dim == 3
        if sample_density is None:
            sample_density = 2 ** self.dim

        self.set_source_velocity(velocity=velocity)

        for i in range(self.dim):
            if translation:
                self.source_bound[0][i] = translation[i]
            else:
                self.source_bound[0][i] = 0

        self.voxelizer.voxelize(triangles)
        t = time.time()
        self.seed_from_voxels(material, color, sample_density)
        ti.sync()
        # print('Voxelization time:', (time.time() - t) * 1000, 'ms')

    @ti.kernel
    def seed_from_external_array(self, num_particles: ti.i32,
                                 pos: ti.ext_arr(), new_material: ti.i32,
                                 color: ti.i32):

        for i in range(num_particles):
            x = ti.Vector.zero(ti.f32, n=self.dim)
            if ti.static(self.dim == 3):
                x = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
            else:
                x = ti.Vector([pos[i, 0], pos[i, 1]])
            self.seed_particle(self.n_particles[None] + i, x, new_material,
                               color, self.source_velocity[None])

        self.n_particles[None] += num_particles

    def add_particles(self,
                      particles,
                      material,
                      color=0xFFFFFF,
                      velocity=None):
        self.set_source_velocity(velocity=velocity)
        self.seed_from_external_array(len(particles), particles, material,
                                      color)

    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.x:
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    @ti.kernel
    def copy_dynamic(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.x:
            np_x[i] = input_x[i]

    @ti.kernel
    def copy_ranged(self, np_x: ti.ext_arr(), input_x: ti.template(),
                    begin: ti.i32, end: ti.i32):
        ti.no_activate(self.particle)
        for i in range(begin, end):
            np_x[i - begin] = input_x[i]

    @ti.kernel
    def copy_ranged_nd(self, np_x: ti.ext_arr(), input_x: ti.template(),
                       begin: ti.i32, end: ti.i32):
        ti.no_activate(self.particle)
        for i in range(begin, end):
            for j in ti.static(range(self.dim)):
                np_x[i - begin, j] = input_x[i, j]

    def particle_info(self):
        np_x = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_x, self.x)
        np_v = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_v, self.v)
        np_material = np.ndarray((self.n_particles[None],), dtype=np.int32)
        self.copy_dynamic(np_material, self.material)
        np_color = np.ndarray((self.n_particles[None],), dtype=np.int32)
        self.copy_dynamic(np_color, self.color)
        return {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'color': np_color
        }

    @ti.kernel
    def clear_particles(self):
        self.n_particles[None] = 0
        ti.deactivate(self.x.loop_range().parent().snode(), [])

    @ti.kernel
    def seed_uniform_cube(self, dx: ti.f32, material: ti.i32,
                          color: ti.i32, cube_size: ti.template(),
                          particles_per_block_length: ti.i32,
                          particles_per_block_length2: ti.i32):
        blocks = ti.Vector.zero(ti.i32, 3)
        for i in ti.static(range(3)):
            blocks[i] = cube_size[i] // particles_per_block_length2
        block2_size = particles_per_block_length2 ** self.dim
        block_size = particles_per_block_length ** self.dim
        length_quotient = particles_per_block_length2 // particles_per_block_length
        n_particles = self.n_particles[None]
        for I in ti.grouped(ti.ndrange(*cube_size)):
            block2_id = I // particles_per_block_length2
            block_id = I % particles_per_block_length2 // particles_per_block_length
            block_index = I % particles_per_block_length
            i = n_particles + ((block2_id[0] * blocks[1] + block2_id[1]) *
                               blocks[2] + block2_id[2]) * block2_size + \
                ((block_id[0] * length_quotient + block_id[1]) *
                 length_quotient + block_id[2]) * block_size + \
                (block_index[0] * particles_per_block_length + block_index[1]) * \
                particles_per_block_length + block_index[2]
            self.seed_particle(i, I * dx + self.source_bound[0], material,
                               color, self.source_velocity[None])

    def add_uniform_cube(self,
                         lower_corner,
                         cube_size,
                         material,
                         color=0xFFFFFF,
                         dx=None,
                         velocity=None):
        if dx is None:
            dx = self.dx / 2
        vol = 1
        for i in range(self.dim):
            vol = vol * cube_size[i]
        num_new_particles = int(vol / dx ** self.dim + 0.5)
        assert self.n_particles[
                   None] + num_new_particles <= self.max_num_particles

        self.set_source_velocity(velocity=velocity)

        particles_per_block_length = 2
        particles_per_block_length2 = 4
        for i in range(self.dim):
            self.source_bound[0][i] = lower_corner[i]
            cube_size[i] = int(cube_size[i] / dx + 0.5)
            assert cube_size[i] % particles_per_block_length2 == 0

        self.seed_uniform_cube(dx, material, color, tuple(cube_size),
                               particles_per_block_length,
                               particles_per_block_length2)
        self.n_particles[None] += num_new_particles

    def init_cube(self):
        t = time.time()
        grid_len = self.res[0]
        assert grid_len == self.res[1]
        assert grid_len == self.res[2]
        if self.use_gpu:
            cube_len = 128  # 128**3 * 8 = 16M
        else:
            cube_len = 64  # 64**3 * 8 = 2M

        cube_left = self.size / 2 - cube_len / 2 * self.dx + 0.25 * self.dx
        # the center of the bottom layer is 4.25dx higher than the ground
        cube_bottom = 12.25 * self.dx
        self.add_uniform_cube(lower_corner=[cube_left, cube_bottom, cube_left],
                              cube_size=[cube_len / grid_len,
                                         cube_len / grid_len,
                                         cube_len / grid_len],
                              material=MPMSolver.material_elastic)
        ti.sync()
        print('init_cube():', time.time() - t, 's', flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',
                        type=str,
                        default='cuda',
                        help='Arch (x64 or cuda)')
    parser.add_argument('-s',
                        '--show',
                        action='store_true',
                        help='Run with gui')
    parser.add_argument('-f',
                        '--frames',
                        type=int,
                        default=2,
                        help='Number of frames')
    parser.add_argument('-r', '--res', type=int, default=256, help='1 / dx')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        help='Prefix of output file name (a binary file for each frame)',
                        default=None)
    parser.add_argument(
        '--no-ad',
        action='store_true',
        help='Disable bit struct atomic demotion',
    )
    parser.add_argument(
        '--no-fusion',
        action='store_true',
        help='Disable bit struct store fusion',
    )
    parser.add_argument(
        '--no-quant',
        action='store_true',
        help='Disable quantization',
    )
    parser.add_argument(
        '--no-g2p2g',
        action='store_true',
        help='Disable g2p2g',
    )
    args = parser.parse_args()
    print(args)
    return args


args = parse_args()

with_gui = False
write_to_disk = args.output is not None
show_timeline = False

use_gpu = (args.arch == 'cuda')

ti.init(arch=ti.cuda if use_gpu else ti.x64,
        kernel_profiler=True,
        use_unified_memory=False,
        device_memory_fraction=0.7,
        quant_opt_store_fusion=not args.no_fusion,
        quant_opt_atomic_demotion=not args.no_ad)

grid_len = args.res
size = 1
dx = size / grid_len

mpm = MPMSolver(res=(grid_len, grid_len, grid_len), size=size, padding=8,
                unbounded=True,
                quant=not args.no_quant,
                use_g2p2g=not args.no_g2p2g,
                support_plasticity=True,
                dt_scale=1.28, E_scale=0.02, vol_scale=1 / 8, nu=0.4,
                use_gpu=use_gpu)
assert mpm.default_dt == 1e-4
assert mpm.p_vol == 2 ** -27
assert mpm.E == 2e4
assert mpm.nu == 0.4
assert mpm.dx == dx
mpm.set_gravity((0, -9.8, 0))

num_frames = args.frames
assert num_frames >= 2
fps = 25
substeps_per_frame = int(1 / fps / mpm.default_dt + 0.1)
assert substeps_per_frame == 400

mpm.init_cube()
for frame in range(num_frames + 1):
    if frame == 2:
        print('First frame:')
        ti.misc.util.print_async_stats(False)
        ti.get_kernel_stats().clear()
        ti.kernel_profiler_clear()
        if show_timeline:
            ti.timeline_clear()
    if frame > 0:
        mpm.step(1 / fps, print_stat=True, add_to_stat=(frame >= 2))
    # colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
    #                   dtype=np.uint32)
    particles = mpm.particle_info()
    np_x = particles['position']
    num_particles = len(np_x)
    # print(f'frame {frame}: {num_particles} particles')
    if write_to_disk:
        filename = f'{args.output}_frame{frame:03d}.bin'
        packed = struct.pack(f'{num_particles * 3}f', *(np_x.reshape(-1)))
        with open(filename, 'wb') as f:
            f.write(packed)

    # simple camera transform
    # screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    # screen_y = (np_x[:, 1])
    #
    # screen_pos = np.stack([screen_x, screen_y], axis=-1)
    #
    # gui.circles(screen_pos, radius=1.5, color=colors[particles['material']])
    # gui.show(f'{frame:06d}.png' if write_to_disk else None)
assert mpm.total_substeps == num_frames * substeps_per_frame
if show_timeline:
    ti.timeline_save('mpm_timeline.json')
print(f'{mpm.total_substeps} substeps completed. '
      f'Last {(num_frames - 1) * substeps_per_frame} frames:')
print(f'  End-to-end time: {mpm.total_time:.4f} s')
print(f'  Total kernel time: {mpm.total_kernel_time:.4f} s')
ti.memory_profiler_print()
