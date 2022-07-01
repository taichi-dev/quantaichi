import taichi as ti


def add_benchmark_args(parser):
    parser.add_argument('-n', type=int, help='Number of repeats', default=10)

    parser.add_argument('-D',
                        '--debug',
                        action='store_true',
                        help='Use debug mode')
    parser.add_argument('--arch',
                        type=str,
                        default='cuda',
                        help='Arch (x64 or cuda)')
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


def extract_init_kwargs(args):
    return {
        'arch': getattr(ti, args.arch),
        'debug': args.debug,
        'quant_opt_store_fusion': not args.no_fusion,
        'quant_opt_atomic_demotion': not args.no_ad,
        'kernel_profiler': True,
        'device_memory_GB': 3
    }
