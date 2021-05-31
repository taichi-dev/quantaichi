import taichi as ti


@ti.func
def tube_domain(dim: ti.template(), x):
    # A ring in 2D and a tube in 3D
    dist_to_center = ti.Vector([x[0], x[1]]).norm()
    if ti.static(dim == 3):
        # Leave some padding at ends of the Z axis
        if not (-0.45 < x[2] < 0.45):
            dist_to_center += 1.0
    return int(0.32 < dist_to_center < 0.45)
