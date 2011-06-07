import setuptools

setuptools.setup(
    name='lmj.kohonen',
    version='1.1',
    py_modules=['kohonen'],
    install_requires=['numpy'],
    author='Leif Johnson',
    author_email='leif@leifjohnson.net',
    description='A small library of Kohonen-style vector quantizers.',
    long_description=open('README').read(),
    license='MIT',
    keywords=('kohonen self-organizing-map neural-gas growing-neural-gas '
              'vector-quantization machine-learning'),
    url='http://github.com/lmjohns3/py-kohonen/',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
