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
        self.norm = np.zeros( (x.shape[1] + 1, 2) )
        self.norm[ 0, 0 ] = np.min( y ) # 目的変数の最小値
        self.norm[ 0, 1 ] = np.max( y ) # 目的変数の最大値
        self.norm[ 1:, 0 ] = np.min( x, axis = 0 ) # 説明変数の最小値
        self.norm[ 1:, 1 ] = np.max( x, axis = 0 ) # 説明変数の最大値

    def normalize(self, x, y=None):
        # nomarize 0 ~ 1 データを0〜1に正規化

        list = self.norm[ 1:,1 ] - self.norm[1:,0]
        list[ list == 0 ] = 1
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

    def fit(self, x, y):
        self.fitnorm( x, y )
        x, y = self.normalize( x, y ) ## データの正規化

        self.bata = np.zeros( ( x.shape[ 1 ] + 1, ) ) #リストの最初の値が y=ax+b のb 続く値がaになる

        for _ in range( self.epoches ):
            for p, q in zip( x, y ):
                z = self.predict( p.reshape( ( 1, -1 ) ), normalized=True )
                z = z.reshape( ( 1, ) )
                err = ( z - q ) * self.lr
                delta = p * err
                self.bata[ 0 ] -= err
                self.bata[ 1: ] -= delta

            ## earlystop が有効な場合
            if self.earlystop is not None:
                z = self.predict(x, nomalized=True )
                s = self.r2( y, z )
                # 値が一定以上になった時点でストップ
                if  self.earlystop <= s:
                    break
        return self

    def predict( self, x, normalized=False ):
        ## 線形回帰モデルの実行
        if not normalized:
            x, _ = self.normalize( x )

        z = np.zeros( ( x.shape[ 0 ], 1 ) ) + self.bata[ 0 ] # y = a_1 * x_1 +  a_2 * x_2 ... + b から誤差を引く

        for i in range( x.shape[ 1 ]):
            c = x[ :, i ] * self.bata[ i + 1 ] # なんかa_i の誤差を反映？
            z += c.reshape( ( -1, 1 ) )

        if not normalized:
            z = z * ( self.norm[ 0, 1 ] - self.norm[ 0, 0 ] ) + self.norm[ 0, 0 ] # 正規化を戻す

        return z

    def __str__( self ):
        if type( self.bata ) is not type( None ):
            s = [ '%f'%self.bata[0] ]
            e = [ ' + feat[ %d ] * %f'%( i+1, j ) for i, j in enumerate( self.bata[ 1: ] ) ]
            s.extend( e )
            return ''.join( s )
        else:
            return '0,0'

if __name__ == '__main__':
    import pandas as pd
    ps = support.get_base_args()
    ps.add_argument('--epoches', '-p', type=int, default=20, help='Num of Epoches')
    ps.add_argument('--leaningrate', '-l', type=float, default=0.01, help='Learning Rate')
    ps.add_argument('--earlystop', '-a', action='store_true', help='Early Stopping')
    ps.add_argument('--stoppingvalue', '-v', type=float, default=0.01, help='Early Stopping')

    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separatar, header=args.header, index_col=args.indexcol)
    x = df[ df.columns[ :-1 ] ].values

    if not args.regression:
        print('Not Support')
    else:
        y = df[ df.columns[ -1 ] ].values.reshape( ( -1, 1 ) )
        if args.earlystop:
            plf = linear(epoches=args.epoches,lr=args.leaningrate, earlystop=args.stoppingvalue)
        else:
            plf = linear(epoches=args.epoches,lr=args.leaningrate)
        support.report_regressor(plf, x, y, args.crossvalidate)



