from distutils.core import setup

setup(
    name='image_tools',
    author='Martin Privat',
    version='0.1',
    packages=['image_tools','image_tools.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='image processing functions',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "opencv-python",
        "scikit-image"
    ]
)