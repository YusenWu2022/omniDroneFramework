"""this file is used to visualize the environment"""
import os

import argparse
import jax

from omnidrone import jumpy as jp
from omnidrone import engine
from omnidrone import envs
from omnidrone.engine.io import html

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--env',
        type=str,
        default='quadrotor_target_dyn',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='./out',
    )
    parser.add_argument(
        '--stastics',
        action='store_true',
    )

    args, _ = parser.parse_known_args()

    # Initialize env
    env = envs.create(args.env)
    state = env.reset(rng=jp.random_prngkey(seed=0))  # set seed

    # mkdir
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Visualize
    if args.stastics:
        # store html to out dir
        htm = html.render(env.sys, [state.qp])

    else:
        @jax.jit
        def f(state, _):
            action = jp.sin(jp.pi / 15 + jp.arange(0, 4) * jp.pi)
            state = env.step(state, action)
            return state, state.qp
        final_state, rollout_qp = jax.lax.scan(f, state, None, length=100)

        # I think this is a more convenient way to transform the data
        # but I am not familiar with jax
        rollout = []
        for i in range(100):
            rollout.append(engine.QP(
                rollout_qp.pos[i], rollout_qp.rot[i], rollout_qp.vel[i], rollout_qp.ang[i]))

        # store html to out dir
        htm = html.render(env.sys, rollout)

    with open(os.path.join(args.out_dir, f'{args.env}.html'), 'w') as f:
        f.write(htm)
