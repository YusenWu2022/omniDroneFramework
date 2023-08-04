Wrappers
========

Wrappers are a way to add functionality to a envionment without modifying the
envionment itself.  This is useful for adding evaluation metrics, vmap, or
other functionality to an environment. Using Wrappers is as simple as
send the envionment to the wrapper and then calling the wrapper as if it
were the envionment. Different wrappers can be combined to create more
complex functionality.

For example, to create an environment that can be vmaped, you can do the
following:

.. code-block:: python

    env = omnidrone.env.make('drone_target')
    env = omnidrone.wrapper.VmapWrapper(env, batch_size=4096)

    state = env.reset(rng=jp.random_prngkey(seed=args.seed))
    print(state.obs.shape) # (4096, env.observation_size)

This will create an environment that can be vmaped. The batch size 
that is can be set to any value 
is the number of environments that will be run in parallel.

After the environment is wrapped, it can be used as if it were the
original environment. You can use the same methods and attributes as
before.

Episode Wrapper
---------------

The EpisodeWrapper is a wrapper that can reste the environment after 
a certaion number of steps which is parmaterized by the max_episode_steps 
variable. This is useful for generate trajectories of a certain length.

.. code-block:: python

    __init__(self, env: Wrapper, max_episode_steps: int, action_repeat: int)

Parameters
    + ``env``: The environment will be wrapped
    + ``max_episode_steps``: The number of steps to run the environment for
    + ``action_repeat``: The number of times to repeat the action

.. Note::
    Brax designed a variable called ``max_episode_steps`` which is the number of
    times to repeat the action, it's not the functionality of the eqisode wrapper.
    I don't know why they did this, I'll find out later.

Vmap Wrapper
------------
The VmapWrapper is a wrapper that can be used to vectorize the environment.
It's a core functionality used to parallelize the environment, which is 
useful for hancing the sample efficiency of the agent. If you use the GPU jax
version, you can use this wrapper to make you envs parallel run on the GPU.

.. code-block:: python

    __init__(self, env: Wrapper, batch_size: int)

Parameters
    + ``env``: The environment will be wrapped
    + ``batch_size``: The number of environments that will be run in parallel

AutoResetWrapper
----------------

When many environments are run in parallel and some of them have finished,
This finished environments will wait for the other environments to finish
before they can be reset. It will cause the sample efficiency to be low.
But you can use the AutoResetWrapper to auto reset the finished environments.

.. code-block:: python

    __init__(self, env: Wrapper)

Parameters
    + ``env``: The environment will be wrapped

.. Note::
    In Brax code base, they store ``first_qp`` and ``first_obs`` in the state, when
    a environment is reset, they will use these two variables to reset the
    environment. I think it will cause our environment can't be reset randomly. 
    I'll modifying this later.

EvalWrapper
-----------

The EvalWrapper is ued to add evaluation metrics to the environment. It will
track episode total reward, episode length, and if the episode is alive. After 
envionment is wrapped, you can use a ``eval_metrics`` which is storeed in the 
``state.info`` to get the evaluation metrics.

.. code-block:: python

    __init__(self, env: Wrapper)

Parameters
    + ``env``: The environment will be wrapped

``eval_metrics`` is a dataclass which contains the following attributes:
    + ``episode_metrics`` Aggregated episode metrics since the beginning of the episode.
    + ``active_episodes`` Boolean vector tracking which episodes are not done yet.
    + ``episode_steps`` Number of steps in each episode.

How to Write your own Wrapper
-----------------------------

To write your own wrapper, you need to inherit the ``Wrapper`` class and
implement the following methods:

.. code-block:: python

    def __init__(self, env: Wrapper):
        self._env = env

In general you will want to store the envionment that is being wrapped
in the ``_env`` attribute. This will allow you to access the envionment
in the methods that you implement.

.. code-block:: python

    def reset(self, rng: jnp.ndarray) -> State:
        return self._env.reset(rng)

You need call the ``reset`` method of the envionment and return the
result in the ``reset`` method of the wrapper. You can modify the
result before returning it if you want to add functionality that you
need.

.. code-block:: python

    def step(self, state: State, action: jnp.ndarray) -> State:
        return self.env.step(state, action)

Don't forget to call the ``step`` method of the envionment and return
the result when you implement the ``step`` method of the wrapper.

We implement the ``__getattr__`` method to allow you to access the
attributes of the envionment that is being wrapped. This allows you to 
strightforwardly use the wrapper as if it were the envionment.
