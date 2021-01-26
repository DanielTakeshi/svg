from setuptools import setup
from setuptools.extension import Extension

ext_modules = [
    Extension('gym_cloth.physics.cloth',   ['gym_cloth/physics/cloth.pyx']   ),
    Extension('gym_cloth.physics.gripper', ['gym_cloth/physics/gripper.pyx'] ),
    Extension('gym_cloth.physics.point',   ['gym_cloth/physics/point.pyx']   ),
]

setup(
    name='svg',
    version='0.0.1',
    author='Emily Denton, Daniel Seita',
    packages=['svg', 'svg.models'],  # no svg.utils, we don't need a separate utils directory
    zip_safe=False,
    annotate=True,
)
