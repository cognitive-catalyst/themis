from setuptools import setup

setup(
    name='themis',
    version='1.0.0',
    packages=['themis'],
    entry_points={
        'console_scripts': ['themis=themis.cli:run'],
    },
    url='https://github.ibm.com/WatsonTooling/data-science',
    license='',
    author='W.P. McNeill',
    author_email='wmcneill@us.ibm.com',
    description='Watson performance analysis toolkit'
)
