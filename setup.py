from setuptools import find_packages, setup

setup(
    name="mlproject",
    version='0.0.1',
    author='Yoro',
    author_email='fallbayeyoro1@gmail.com',
    packages= find_packages(),
    requires=['pandas', 'numpy']

)