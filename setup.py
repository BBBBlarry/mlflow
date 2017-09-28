from distutils.core import setup

setup(name='mlflow',
      version='0.01',
      author='Zeyu Wang',
      author_email='zwan48@uw.edu',
      packages = ['mlflow', 'mlflow.Exceptions', 'mlflow.Ops', 'mlflow.Train', 'mlflow.utils']
      )