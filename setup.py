try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {

}

setup(description='OpenEphys IO tools',
      author='Ronny Eichler',
      author_email='ronny.eichler@gmail.com',
      version='0.1.2dev',
      license='MIT',
      install_requires=['nose', 'numpy', 'six'],
      packages=['oio'],
      package_data={'config': ['config/*.*']},
      include_package_data=True,
      name='oio',
      entry_points="""
        [console_scripts]
        oio=oio.__main__:main
        get_needed_channels=oio.__main__:get_needed_channels
      """)
