import taichi as ti

# An abstraction of a continuous physical field


@ti.data_oriented
class SparseField:
    def __init__(self,
                 dtype,
                 inv_dx,
                 res,
                 offset,
                 vector_width=None,
                 field=None):
        self.dtype = dtype
        self.dim = len(offset)
        self.stagger = ti.Vector([0.5] * self.dim)
        # TODO: rename res to shape
        self.res = res
        self.shape = res
        self.offset = tuple(offset)
        self.inv_dx = inv_dx
        self.dx = 1 / self.inv_dx

        if field is not None:
            assert vector_width is None
            # assert field.dtype == dtype
            self.field = field
        elif vector_width is None:
            self.field = ti.field(dtype=dtype)
        else:
            self.field = ti.Vector.field(vector_width, dtype=dtype)

    @staticmethod
    @ti.func
    def lerp(vl, vr, frac):
        # frac: [0.0, 1.0]
        return vl + frac * (vr - vl)

    # TODO: support operator overloading?

    @ti.func
    def bounded_load(self, I):
        # TODO: assume self.inv_dx is grid size
        # get rid of this after the simulation is unbounded
        for d in ti.static(range(I.n)):
            I[d] = max(self.offset[d],
                       min(self.res[d] + self.offset[d] - 1, I[d]))
        return self.field[I]

    @ti.func
    def store(self, I, val):
        self.field[I] = val

    @ti.func
    def _sample_abcd(self, I):
        Ip = I  # I prime
        a = self.bounded_load(Ip)  # +(0, 0)
        Ip[0] += 1
        b = self.bounded_load(Ip)  # +(1, 0)
        Ip = I
        Ip[1] += 1
        c = self.bounded_load(Ip)  # +(0, 1)
        Ip[0] += 1
        d = self.bounded_load(Ip)  # +(1, 1)
        return a, b, c, d

    @ti.func
    def _sample2d(self, I, fu, fv):
        a, b, c, d = self._sample_abcd(I)
        return self.lerp(self.lerp(a, b, fu), self.lerp(c, d, fu), fv)

    @ti.func
    def sample(self, p):
        uv = p * self.inv_dx
        st = uv - self.stagger
        D = ti.static(self.dim)
        iuv = ti.Vector.zero(ti.i32, D)
        fuv = ti.Vector.zero(ti.f32, D)
        for d in ti.static(range(D)):
            s = st[d]
            iu = int(ti.floor(s))
            iuv[d] = iu
            fuv[d] = s - iu

        sp1 = self._sample2d(iuv, fuv[0], fuv[1])
        if ti.static(D == 3):
            iuv[2] += 1
            sp2 = self._sample2d(iuv, fuv[0], fuv[1])
            sp1 = self.lerp(sp1, sp2, fuv[2])
        return sp1

    @ti.func
    def _sample2d_minmax(self, I):
        a, b, c, d = self._sample_abcd(I)
        return min(a, b, c, d), max(a, b, c, d)

    @ti.func
    def sample_minmax(self, p):
        uv = p * self.inv_dx
        st = uv - self.stagger
        D = ti.static(self.dim)
        iuv = ti.Vector.zero(ti.i32, D)
        for d in ti.static(range(D)):
            s = st[d]
            iuv[d] = int(ti.floor(s))

        min1, max1 = self._sample2d_minmax(iuv)
        if ti.static(D == 3):
            iuv[2] += 1
            min2, max2 = self._sample2d_minmax(iuv)
            min1 = min(min1, min2)
            max1 = max(max1, max2)

        return min1, max1

    @ti.func
    def I2p(self, I):
        return (I + self.stagger) * self.dx

    def create_decomposed_sparse_fields(self):
        assert isinstance(self.field, ti.Matrix)
        vector_width = self.field.n
        self.decomponsed_sparse_fields = [
            SparseField(
                self.dtype,
                self.inv_dx,
                self.res,
                self.offset,
                field=self.field(i)) for i in range(vector_width)
        ]

    def get_sparse_field_at(self, i):
        return self.decomponsed_sparse_fields[i]


if __name__ == '__main__':
    ti.init(debug=True)
    N = 128
    a = SparseField(dtype=ti.f32, inv_dx=N)

    ti.root.dense(ti.ij, N).place(a.field)

    @ti.kernel
    def foo():
        a.store(ti.Vector([1, 1]), 1.0)
        assert abs(a.sample(ti.Vector([1.5 / N, 1.5 / N])) - 1.0) < 1e-6
        assert abs(a.sample(ti.Vector([1.5 / N, 1 / N])) - 0.5) < 1e-6
        assert abs(a.sample(ti.Vector([1.5 / N, 0.5 / N]))) < 1e-6

    foo()
