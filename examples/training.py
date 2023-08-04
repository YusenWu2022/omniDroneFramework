import os

import wandb
import argparse

from omnidrone import jumpy as jp
from omnidrone import envs
from omnidrone.training import algos
from omnidrone.algos.utils.cfg import get_kwargs_from_yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        default='ppo',
        help='Algorithm to use for training. Options are: ppo, apg, ars, es, sac, shac, ppol, apgl'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='quadrotor_target_dyn',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for training.'
    )
    parser.add_argument(
        '--CUDA_VISIBLE_DEVICES',
        type=str,
        default='0',
        help='CUDA_VISIBLE_DEVICES'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Use wandb for logging.'
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default='./cfgs/ppo.yaml',
        help='Path to config file.'
    )
    args, _ = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # Initialize wandb
    if args.wandb:
        wandb.init(entity="pku_rl", project="OmniDrone", name="ppol_jax")

    # Initialize env
    env = envs.get_environment(args.env)
    state = env.reset(rng=jp.random_prngkey(seed=args.seed))  # set seed

    # Initialize algorithm
    kwargs = get_kwargs_from_yaml(args.algo, args.env, args.cfg)
    algo = algos[args.algo].train(env, **kwargs)
