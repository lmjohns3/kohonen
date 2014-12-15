import os
import setuptools

setuptools.setup(
    name='kohonen',
    version='1.1.2',
    packages=setuptools.find_packages(),
    author='Leif Johnson',
    author_email='leif@leifjohnson.net',
    description='A library of vector quantizers',
    long_description=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='http://github.com/lmjohns3/py-kohonen',
    keywords=('kohonen '
              'self-organizing-map '
              'neural-gas '
              'growing-neural-gas '
              'vector-quantization '
              'machine-learning'),
    install_requires=['numpy'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
