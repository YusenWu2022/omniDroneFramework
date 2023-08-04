from setuptools import find_packages
from setuptools import setup

setup(
    name="omnidrone",
    version="0.0.1",
    description=("omnidrone is a differentiable physics engine that simulates environments made up of rigid bodies, joints, and actuators for safe reinforcement learning."),
    author="PKU-MARL",
    author_email="yaodong.yang@pku.edu.cn",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    # scripts=["bin/learn"],
    install_requires=[
        "absl-py>=1.3.0",
        "dataclasses>=0.6",
        "dm_env>=1.5",
        "flax>=0.6.1",
        "gym>=0.26.2",
        "wandb>=0.13.5",
        "grpcio>=1.27.2",
        # "jax==0.3.24",
        # "jaxlib==0.3.24",
        "numpy>=1.23.3",
        "optax>=0.1.3",
        "Pillow>=9.3.0",
        "pytinyrenderer>=0.0.13",
        "tensorboardX>=2.5.1",
        "trimesh>=3.16.0",
        "typing-extensions>=4.4.0",
        "PyYAML>=6.0",
        "ipdb>=0.13.9",
        "IPython>=8.6.0"
    ],
    extras_require={
        "develop": ["pytest", "transforms3d"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="JAX reinforcement learning rigidbody physics"
)
