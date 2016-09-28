from distutils.core import setup
import py2exe
import matplotlib

setup(console = ['classify.py'],
      data_files = matplotlib.get_py2exe_datafiles(),
      options = dict(
          py2exe = dict(
              compressed = True,
              optimize = 2,
              dll_excludes = ['MSVCP90.dll'],
              packages = ['FileDialog'],
              includes = [
                  'scipy',
                  'scipy.integrate',
                  'scipy.special.*',
                  'scipy.linalg.*',
                  'scipy.sparse.csgraph._validation',
                  'matplotlib',
                  'matplotlib.backends.backend_tkagg',
                  'matplotlib.pyplot',
                  'sklearn',
                  'sklearn.pipeline',
                  'sklearn.tree',
                  'sklearn.tree._utils',
                  'sklearn.utils.lgamma'
              ],
              excludes = [
                  '_gtkagg',
                  '_tkagg',
                  '_agg2',
                  '_cairo',
                  '_cocoaagg',
                  '_fltkagg',
                  '_gtk',
                  '_gtkcairo',
                  'tcl'
              ]
          )
      ))
