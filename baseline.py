import re
import numpy as np
import pandas as pd
import support
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

if __name__ == '__main__':
  models = [
    ( 'SVM', SVC( random_state=1 ), SVR()),
    ( 'GaussianProcess',
      GaussianProcessClassifier(random_state=1),
      GaussianProcessRegressor( normalize_y=True, alpha=1, random_state=1)),
    ( 'KNeighbors', KNeighborsClassifier(), KNeighborsRegressor() ),
    ('MLP',
     MLPClassifier( random_state=1 ),
     MLPRegressor( hidden_layer_sizes=(5), solver='lbfgs', random_state=1))
  ]
  #print(models)

  classfier_files = [ 'iris.data', 'sonar.all-data', 'glass.data' ]
  classfier_params = [( ',', None, None), ( ',',None, None), ( ',', None, 0) ]
  regressor_files = [ 'airfoil_self_noise.dat', 'winequality-red.csv','winequality-white.csv' ]
  regressor_params = [ (r'\t', None, None), (';', 0, None), ( ';', 0, None) ]

  result = pd.DataFrame(
    columns=[ 'target', 'function' ] + [ m[0] for m in models],
    index=range(len( classfier_files + regressor_files ) * 2))

  ncol = 0
  for i, (c,p) in enumerate( zip( classfier_files, classfier_params) ):
    # read file
    df = pd.read_csv( c, sep=p[0], header=p[1], index_col=p[2])
    x = df[ df.columns[ :-1 ] ].values

    y, clz = support.clz_to_prob( df[ df.columns[ -1 ] ])

    # create scores
    result.loc[ncol, 'target'] = re.split( r'[._]', c )[0]
    result.loc[ncol + 1, 'target'] = ''
    result.loc[ncol, 'function'] = 'F1 Score'
    result.loc[ncol + 1, 'function'] = 'Accuracy'

    ## write algorithm score
    for l, c_m, r_m in models:
      kf = KFold( n_splits=5, random_state=1, shuffle=True)
      s = cross_validate( c_m, x, y.argmax( axis=1 ), cv=kf, scoring=( 'f1_weighted' , 'accuracy'))
      result.loc[ ncol, l ] = np.mean( s[ 'test_f1_weighted' ] )
      result.loc[ ncol + 1, l ] = np.mean( s[ 'test_accuracy' ] )

    ncol += 2

  for i, (c,p) in enumerate( zip( regressor_files, regressor_params ) ):
    # read file
    df = pd.read_csv( c, sep=p[0], header=p[1], index_col=p[2])
    x = df[ df.columns[ :-1 ] ].values
    y = df[ df.columns[ -1 ] ].values.reshape( (-1, ) )

    # create scores
    result.loc[ncol, 'target'] = re.split( r'[._]', c )[0]
    result.loc[ncol + 1, 'target'] = ''
    result.loc[ncol, 'function'] = 'R2 Score'
    result.loc[ncol + 1, 'function'] = 'MeanSquared'

    ## write algorithm score
    for l, c_m, r_m in models:
      kf = KFold( n_splits=5, random_state=1, shuffle=True)
      s = cross_validate( r_m, x, y, cv=kf, scoring=( 'r2', 'neg_mean_squared_error' ))
      result.loc[ ncol, l ] = np.mean( s[ 'test_r2' ] )
      result.loc[ ncol + 1, l ] = np.mean( s[ 'test_neg_mean_squared_error' ] )

    ncol += 2

  print(result)
  result.to_csv('baseline.csv',index=None)

