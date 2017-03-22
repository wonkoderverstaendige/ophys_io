try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import oio

setup(description='OpenEphys IO tools',
      author='Ronny Eichler',
      author_email='ronny.eichler@gmail.com',
      version=oio.__version__,
      license='MIT',
      install_requires=['nose', 'numpy', 'six', 'click', 'tqdm'],
      packages=['oio'],
      package_data={'config': ['config/*.*']},
      include_package_data=True,
      name='oio',
      entry_points="""
        [console_scripts]
        oio=oio.__main__:main
        get_needed_channels=oio.util:get_needed_channels
      """)
