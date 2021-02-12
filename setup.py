from setuptools import setup

setup(name='DeepQuantreg',
      version='0.2',
      description='Deep Censored Quantile Regression',
      url='http://github.com/yij22/DeepQuantreg',
      author='Yichen Jia',
      author_email='yij22@pitt.edu',
      license='MIT',
      packages=['DeepQuantreg'],
      install_requires=[
          'pandas','numpy','tensorflow','keras',
          'lifelines','sklearn','scipy'
      ],
      zip_safe=False)
