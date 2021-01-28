from setuptools import setup, find_packages

setup(
    version='0.0.1',
    name="vptree",
    description='A python implementation of Vantage Point Tree',
    py_modules=['vptree'],
    packages=find_packages(where='vptree'),
    install_requires=['numpy']
)