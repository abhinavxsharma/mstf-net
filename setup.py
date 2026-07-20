from setuptools import setup, find_packages

setup(
    name='mstfnet',
    version='1.0.0',
    author='Abhinav Vats',
    author_email='vats.abhinav247@gmail.com',
    description='MSTF-Net: Adaptive Multi-Stream Deepfake Detection via Dynamic Spectral-Temporal Gating',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/abhinavxsharma/mstf-net',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.4.0',
        'torchvision>=0.19.0',
        'timm==0.9.16',
        'opencv-python-headless>=4.9.0',
        'scikit-learn>=1.4.0',
        'pyyaml>=6.0',
        'numpy>=1.26.0',
        'pillow>=10.3.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='deepfake detection, multi-stream, SRM, DSTG, computer vision',
)