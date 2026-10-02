PRESETS = {
    'strategy_u': {
        'lr': 0.004,
        'l2_threshold': 0.2,
        'early_stop_gate': 0.5,
        'early_stop_target': 0.5,
        'max_no_improve': 5,
        'patience': 5,
    },
    'strategy_p': {
        'lr': 0.0025,
        'l2_threshold': 0.2,
        'early_stop_gate': 0.6,
        'early_stop_target': 0.6,
        'max_no_improve': 5,
        'patience': 5,
    },
    'biased': {
        'lr': 0.0025,
        'l2_threshold': 0.95,
        'early_stop_gate': 0.95,
        'early_stop_target': 0.98,
        'max_no_improve': 5,
        'patience': 5,
    },
}


def apply_preset(args):
    preset = PRESETS[args.mode]

    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.threshold is not None:
        args.l2_threshold = args.threshold
    else:
        args.threshold = args.l2_threshold

    return args
