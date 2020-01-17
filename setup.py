from distutils.core import setup

REQUIRED_PKGS = ['tensorflow==2.0.0', 'scikit-learn==0.22.1']

TESTS_REQUIRE = ['pytest==5.2.2', 'coverage==5.0.3']

setup(
    name='TensorFM',
    version='1.0.0',
    packages=['tensorfm',],
    license='MIT',
    long_description=open('README.md').read(),
)
