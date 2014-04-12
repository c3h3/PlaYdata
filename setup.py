#try:
#    from setuptools import setup
#except ImportError:
#    from distutils.core import setup

from setuptools import setup, find_packages

setup(
    name='PlaYdata',
    version='0.0.1dev',
    description='',
    author='Chia-Chi Chang & Willy Kuo',
    author_email='c3h3.tw@gmail.com & waitingkuo0527@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        "pandas"
    ],
)
