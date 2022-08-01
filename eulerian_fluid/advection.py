import taichi as ti


@ti.func
def lerp(vl, vr, frac):
    return vl + frac * (vr - vl)


@ti.data_oriented
class AdvectionOp:
    def __init__(self, rk_order=3, advect_op='mc'):
        self.backtrace = self.backtrace_operators[rk_order - 1]
        if advect_op == 'sl':
            self.advect_operator = self.advect_sl
        elif advect_op == 'mc':
            self.advect_operator = self.advect_mc
        else:
            raise ValueError(f'Unknown error {advect_op}')

    def advect(self, *args):
        self.advect_operator(*args)

    @ti.func
    def backtrace_rk1(v: ti.template(), p, dt):
        p -= dt * v.sample(p)
        return p

    @ti.func
    def backtrace_rk2(v: ti.template(), p, dt):
        p_mid = p - 0.5 * dt * v.sample(p)
        p -= dt * v.sample(p_mid)
        return p

    @ti.func
    def backtrace_rk3(v: ti.template(), p, dt):
        v1 = v.sample(p)
        p1 = p - 0.5 * dt * v1
        v2 = v.sample(p1)
        p2 = p - 0.75 * dt * v2
        v3 = v.sample(p2)
        r = p - dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
        return r

    backtrace_operators = [backtrace_rk1, backtrace_rk2, backtrace_rk3]

    @ti.kernel
    def advect_sl(self, v: ti.template(), q_in: ti.template(),
                  q_out: ti.template(), q_tmp: ti.template(), dt: ti.f32):
        # Semi-Lagrangian
        for I in ti.grouped(q_in.field):
            p = q_in.I2p(I)
            p = self.backtrace(v, p, dt)
            q_out.field[I] = q_in.sample(p)

    @ti.kernel
    def advect_mc(self, v: ti.template(), q_in: ti.template(),
                  q_out: ti.template(), q_tmp: ti.template(), dt: ti.f32):
        # MacCormack
        for I in ti.grouped(q_in.field):
            p = q_in.I2p(I)
            p = self.backtrace(v, p, dt)
            q_out.field[I] = q_in.sample(p)

        for I in ti.grouped(q_in.field):
            p = q_in.I2p(I)
            p = self.backtrace(v, p, -dt)
            q_tmp.field[I] = q_out.sample(p)

        for I in ti.grouped(q_in.field):
            p = q_in.I2p(I)
            p_src = self.backtrace(v, p, dt)
            p_sl = q_in.sample(p_src)
            q_out.field[I] = q_out.field[I] + 0.5 * (q_in.field[I] - q_tmp.field[I])

            min_val, max_val = q_in.sample_minmax(p_src)
            cond = min_val < q_out.field[I] < max_val
            if ti.static(isinstance(cond, ti.Matrix)):
                for k in ti.static(range(cond.n)):
                    if not cond[k]:
                        q_out.field[I][k] = p_sl[k]
            else:
                if not cond:
                    q_out.field[I] = p_sl
