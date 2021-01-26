from setuptools import setup

setup(
    name='svg',
    version='0.0.1',
    author='Emily Denton, Daniel Seita',
    packages=['svg', 'svg.models'],  # no svg.utils, we don't need a separate utils directory
    zip_safe=False,
    annotate=True,
)
