from setuptools import setup
# Note: setup() has access to cmd arguments of the setup.py script via sys.argv

setup(name="AADL",
      version='1.0',
      package_dir={'AADL': 'AADL'},
      packages=['AADL'],
      python_requires='>=3.11',
      install_requires=[
          'torch>=2.13',
          'numpy>=2.0',
      ],
      author='Massimiliano Lupo Pasini, Viktor Reshniak, Miroslav Stoyanov',
      author_email='lupopasinim@ornl.gov',)
