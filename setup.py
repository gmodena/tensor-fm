from setuptools import setup, find_packages

REQUIRED_PKGS = ['tensorflow==2.5.1', 'tensorflow-estimator==2.5.0', 'scikit-learn==0.22.1']
TESTS_REQUIRE = ['pytest==5.2.2']

setup(
    name='TensorFM',
    version='1.0.4',
    author='Gabriele Modena',
    author_email='gm@nowave.it',
    packages=find_packages(),
    license='MIT',
    install_requires=REQUIRED_PKGS,
    tests_require=TESTS_REQUIRE,
    long_description=open('README.md').read(),
)
