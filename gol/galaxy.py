import taichi as ti
import numpy as np
import argparse
import os
import gzip
from datetime import datetime


def from_rle(file):
    width = -1
    height = -1
    data = np.zeros((4, 4), dtype=np.uint8)
    x, y, n = 0, 0, 0
    count = 0
    with open(file, "r") as f:
        for line in f:
            if line[0] == '#':
                # skip comment lines
                continue
            elif line[0] == 'x':
                # read params
                tokens = line.split(',')
                width = int(tokens[0][4:])
                height = int(tokens[1][4:])
                data = np.zeros((height, width), dtype=np.uint8)
            else:
                # process RLE-encoded pattern
                for c in line:
                    if c.isdigit():
                        n = n * 10 + int(c)
                    else:
                        if n == 0:
                            n = 1
                        if c == 'b':
                            for j in range(x, x + n):
                                data[y][j] = 0
                            x += n
                        elif c == 'o':
                            for j in range(x, x + n):
                                data[y][j] = 1
                            x += n
                        elif c == '$':
                            x = 0
                            y += n
                        elif c == '!':
                            break
                        n = 0
    return data, width, height


def from_npy(file):
    with gzip.open(file, 'rb') as f:
        data = np.load(f)
        height, width = data.shape
        return data, width, height


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--show', action='store_true', help='Run with gui')
    parser.add_argument('-a', '--arch', type=str, default='cpu', help='[cpu/cuda]')
    parser.add_argument('-o', '--out-dir', type=str, help='Output folder')
    parser.add_argument('--use-rle-file', action='store_true', help='Read pattern from RLE file')
    parser.add_argument('--steps-per-capture', type=int, default=32768, help='Number of steps per capture')
    args = parser.parse_args()
    print(args)
    return args


args = parse_args()

output_folder = None
if args.out_dir:
    output_folder = f'{args.out_dir}'
    os.makedirs(output_folder, exist_ok=True)
if args.arch == 'cuda':
    ti.init(arch=ti.cuda)
else:
    ti.init(arch=ti.cpu)

qu1 = ti.types.quant.int(1, False)

state_a = ti.field(dtype=qu1)
state_b = ti.field(dtype=qu1)
img_size = 512
N = 65536
bits = 32
n_blocks = 16
n = 30720
boundary_offset = int((N - n) / 2)
MAX_FRAMES = 16

block = ti.root.pointer(ti.ij, (n_blocks, n_blocks))
block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(
    ti.j, bits, max_num_bits=bits).place(state_a)
block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(
    ti.j, bits, max_num_bits=bits).place(state_b)

img = ti.field(dtype=ti.f32, shape=(img_size, img_size))


@ti.kernel
def evolve(x: ti.template(), y: ti.template()):
    ti.loop_config(bit_vectorize=True)
    for i, j in x:
        num_active_neighbors = ti.u32(0)
        num_active_neighbors += ti.cast(x[i - 1, j - 1], ti.u32)
        num_active_neighbors += ti.cast(x[i - 1, j], ti.u32)
        num_active_neighbors += ti.cast(x[i - 1, j + 1], ti.u32)
        num_active_neighbors += ti.cast(x[i, j - 1], ti.u32)
        num_active_neighbors += ti.cast(x[i, j + 1], ti.u32)
        num_active_neighbors += ti.cast(x[i + 1, j - 1], ti.u32)
        num_active_neighbors += ti.cast(x[i + 1, j], ti.u32)
        num_active_neighbors += ti.cast(x[i + 1, j + 1], ti.u32)
        y[i, j] = (num_active_neighbors == 3) | ((num_active_neighbors == 2) & (x[i, j] == 1))


@ti.func
def fill_pixel(scale, buffer, i, j, region_size):
    ii = i * 1.0 / img_size
    jj = j * 1.0 / img_size
    ret_val = 0.0
    if scale > 1:
        sx1, sx2, sy1, sy2 = j * scale, (j + 1) * scale, i * scale, (i + 1) * scale
        x1 = ti.cast(sx1, ti.i32)
        x2 = ti.cast(sx2, ti.i32) + 1
        y1 = ti.cast(sy1, ti.i32)
        y2 = ti.cast(sy2, ti.i32) + 1
        count = 0
        val = 0
        for mm in range(y1, y2):
            for nn in range(x1, x2):
                if mm + 0.5 > sy1 and mm + 0.5 < sy2 and nn + 0.5 > sx1 and nn + 0.5 < sx2:
                    count += 1
                    val += buffer[boundary_offset + int(n / 2) - int(region_size / 2) + mm,
                                  boundary_offset + int(n / 2) - int(region_size / 2) + nn]
        ret_val = val
    else:
        ret_val = buffer[int(boundary_offset + int(n / 2) - region_size / 2 + region_size * ii),
                   int(boundary_offset + int(n / 2) - region_size / 2  + region_size * jj)]
    return ret_val


@ti.kernel
def fill_img(region_size: ti.i32, buffer: ti.template()):
    scale = region_size * 1.0 / img_size
    for i, j in ti.ndrange(img_size, img_size):
        img[i, j] = fill_pixel(scale, buffer, i, j, region_size)


@ti.kernel
def init_randomized(x: ti.template(), y: ti.template()):
    for i in range(boundary_offset, N - boundary_offset):
        for j in range(boundary_offset, N - boundary_offset):
            x[i, j] = ti.cast(ti.floor(ti.random() * 2), ti.u32)
            y[i, j] = 0

@ti.kernel
def clear(x: ti.template(), y: ti.template()):
    for i in range(boundary_offset, N - boundary_offset):
        for j in range(boundary_offset, N - boundary_offset):
            x[i, j] = 0
            y[i, j] = 0

@ti.kernel
def init_from_slices(x: ti.template(), y: ti.template(), init_buffer: ti.types.ndarray(),
                     init_width: ti.i32, offset: ti.i32, rows: ti.i32):
    for i in range(boundary_offset + offset, boundary_offset + offset + rows):
        for j in range(boundary_offset, boundary_offset + init_width):
            x[i, j] = init_buffer[i - boundary_offset - offset, j - boundary_offset]


gui = ti.GUI('Quantized Game of Life', (img_size, img_size), show_gui=args.show)
frame_id = 0

def init_states(init_buffer, init_width, init_height):
    clear(state_a, state_b)
    slice_size = 1000000
    num_rows = slice_size // init_width
    for i in range(0, init_height, num_rows):
        rows = min(num_rows, init_height - i)
        init_from_slices(state_a, state_b, init_buffer[i:i+num_rows, :], init_width, i, rows)

def running(x, y, _gui):
    global frame_id
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    steps = args.steps_per_capture
    print(f"before {steps} steps ", current_time)
    for _ in range(steps // 2):
        evolve(x, y)
        evolve(y, x)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"after {steps} steps ", current_time)
    fill_img(n, x)
    _gui.set_image(ti.tools.imresize(img, img_size, img_size).astype(np.float32))
    _gui.show(f'{output_folder}/{frame_id:06d}.png')
    frame_id += 1


def main():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("pattern start ", current_time)
    if args.use_rle_file:
        init_buffer, init_width, init_height = from_rle('metapixel-galaxy.rle')
        init_buffer = np.rot90(init_buffer, 3)
    else:
        init_buffer, init_width, init_height = from_npy('metapixel-galaxy.npy.gz')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("pattern complete ", current_time)
    init_states(init_buffer, init_width, init_height)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("init complete ", current_time)
    while gui.running and frame_id < MAX_FRAMES:
        running(state_a, state_b, gui)
        running(state_b, state_a, gui)


if __name__ == "__main__":
    main()
