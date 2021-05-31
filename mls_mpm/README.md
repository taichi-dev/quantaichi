# Quantized MLS-MPM Simulation

## Demo

1. See the output of `python3 -m demo.demo_quantized_simulation_letters --help` to learn about how to reproduce the large-scale quantized simulation letter demo. `-s` for a quick visualization. You may need to wait for 30 frames to see letters falling down.

2. Similarly, use `python3 -m demo.demo_3d_bunnies --help` for the flood + bunny demo.

## Benchmark

How to run:
```
cd benchmark
python3 quan_mpm_benchmark.py [--no-ad] [--no-fusion] [--no-quant] [--no-g2p2g]
```

See the output of `cd benchmark && python3 quan_mpm_benchmark.py --help` for more options.
