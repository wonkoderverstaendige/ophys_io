try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'OpenEphys IO tools',
    'author': 'Ronny Eichler',
    'download_url': '',
    'author_email': 'ronny.eichler@gmail.com',
    'version': '0.1.1',
    'license': 'MIT',
    'install_requires': ['nose', 'numpy', 'six'],
    'packages': ['oio'],
    'scripts': [],
    'name': 'oio',
    'entry_points': """
        [console_scripts]
        oio=oio.__main__:main
    """
    }

setup(**config)
