from distutils.core import setup

setup(
    name='image_tools',
    python_requires='>=3.9,<3.10',
    author='Martin Privat',
    version='0.2.9',
    packages=['image_tools','image_tools.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='image processing functions',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "opencv-contrib-python-rolling @ https://github.com/ElTinmar/build_opencv/raw/main/opencv_contrib_python_rolling-4.8.0.20231215-cp39-cp39-linux_x86_64.whl",
        "scikit-image",
        "PyQt5",
        "geometry @ git+https://github.com/ElTinmar/geometry.git@main",
        "qt_widgets @ git+https://github.com/ElTinmar/qt_widgets.git@main"
    ],
    extras_require = {
        'gpu': ["cupy", "cucim"]
    }
)
