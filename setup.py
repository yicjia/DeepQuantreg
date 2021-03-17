from setuptools import setup

setup(name='DeepQuantreg',
      version='0.1',
      description='Deep Learning for Quantile Regression under Right Censoring',
      url='https://github.com/yicjia/DeepQuantreg',
      author='Yichen Jia',
      author_email='yij22@pitt.edu',
      license='MIT',
      packages=['DeepQuantreg'],
      install_requires=[
          'pandas','numpy','tensorflow','keras','sklearn','lifelines','scipy'
      ],
      zip_safe=False)
