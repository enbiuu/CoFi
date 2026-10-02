from importlib import import_module

from utils.util import parse_args
from utils.experiment_presets import apply_preset


RUNNERS = {
    'strategy_u': 'train.main_strategy_u',
    'strategy_p': 'train.main_strategy_p',
    'biased': 'train.main_biased',
}


def main():
    args = apply_preset(parse_args())

    if args.mode not in RUNNERS:
        raise ValueError(f"不支持的 mode: {args.mode}")

    module = import_module(RUNNERS[args.mode])
    run = module.run

    print(f"\n========== 开始运行 {args.mode} | Seed = {args.seed} ==========" )
    run(args)


if __name__ == "__main__":
    main()
