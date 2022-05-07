# install using 'pip install -e .'

from setuptools import setup

setup(name='pointnet',
      packages=['pointnet'],
      package_dir={'pointnet': 'Training'},
      install_requires=['torch',
                        'tqdm',
                        'plyfile',
                        'pyvista'],
    version='0.0.1')