import numpy as np
import support

class linear:
    def __init__(self, epoches=20, lr=0.01, earlystop=None):
        self.epoches = epoches
        self.lr = lr
        self.earlystop = earlystop
        self.beta = None
        self.norm = None

    def fitnorm(self,x,y):
        self.norm = np.zeros( x.shape[1] + 1, 2 )
        self.norm[ 0, 0 ] = np.min( y )
        self.norm[ 0, 1 ] = np.max( y )
        self.norm[ 1:, 0 ] = np.min( x, axis = 0 )
        self.norm[ 1:, 1 ] = np.min( x, axis = 0 )

    def nomarize(self, x, y=None):
        # nomarize 0 ~ 1

        list = self.norm[ 1:,1 ] - self.norm[1:,0]
        list[ list == 0 ] = list
        p = ( x - self.norm[ 1:,0 ] ) / list

        q = y
        if y is not None and not self.norm[ 0,1 ] == self.norm[ 0,0 ]:
            q = ( y - self.norm[ 0,0 ] ) / ( self.norm[ 0,1 ] - self.norm[ 0,0 ] )

        return p,q

    def r2(self, y, z):

        y = y.reshapre( ( -1, ) )
        z = z.reshapre( ( -1, ) )

        mn = ( ( y - z ) ** 2 ).sum( axis=0 )
        dn = ( ( y - y.mean() ) ** 2 ).sum( axis=0 )

        if dn == 0:
            return np.inf

        return 1 - mn / dn

    #def fit(self, x, y):
