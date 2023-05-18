#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='oai_agents',
      version='0.1.0',
      description='Simple implementations of transformers to encode and decode most modalities',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Stéphane Aroca-Ouellette',
      author_email='stephanearocaouellette@gmail.com',
      url='https://github.com/StephAO/simple_transformers',
      download_url='https://github.com/StephAO/simple_transformers',
      keywords=['Transformers', 'Deep Learning', 'Machine Learning'],
      packages=['simple_transformers'],
      package_dir={
          'simple_transformers': 'simple_transformers',
      },
      package_data={
          'simple_transformers': [
              'simple_transformers.config.yaml'
          ],
      },
      install_requires=[
        'numpy',
        'transformers',
        'datasets'
      ],
      tests_require=['pytest']
    )