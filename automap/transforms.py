
import numpy as np

def from_json(js):
    cls = {'Polynomial':Polynomial,
           'Projection':Projection,
           'TIN':TIN,
           'Chain':Chain,
           }[js['type']]
    trans = cls.from_json(js)
    return trans

class Chain(object):

    def __init__(self, transforms=None):
        '''A chain of multiple transforms executed consecutively'''
        self.transforms = [t for t in transforms] if transforms else []

    def info(self):
        # TODO: rename to_json()
        params = {}
        data = {'transforms':[trans.info() for trans in self.transforms] }
        info = {'type': 'Chain',
                'params': params,
                'data': data,
                }
        return info

    @staticmethod
    def from_json(js):
        print 'chain fromjs', js
        init = {}
        transforms = [from_json(transdict) for transdict in js['data']['transforms']]
        init['transforms'] = transforms
        init.update(js['params'])
        trans = Chain(**init)
        return trans

    def add(self, transform):
        self.transforms.append(transform)

    def predict(self, x, y):
        for trans in self.transforms:
            x,y = trans.predict(x, y)
        return x,y

class Polynomial(object):
    
    def __init__(self, order=None, A=None):
        '''Polynomial transform of 1st (affine), 2nd, or 3rd order'''
        if A is not None:
            A = np.array(A)
            if A.shape == (3,3):
                order = 1
            elif A.shape == (6,6):
                order = 2
            elif A.shape == (10,10):
                order = 3
            else:
                raise ValueError('Matrix A must be shape (3,3), (6,6), or (10,10); not {}'.format(A.shape))

        self.A = A
        self.order = order
        self.minpoints = {1:3, 2:10, 3:20}.get(order, 3) # minimum 3 if order not set

    def __repr__(self):
        return u'Polynomial Transform(order={}, estimated={})'.format(self.order, self.A is not None)

    def copy(self):
        new = Polynomial(order=self.order, A=self.A)
        new.minpoints = self.minpoints
        return new

    def info(self):
        # TODO: rename to_json()
        params = {'order': self.order}
        data = {'A': self.A.tolist() }
        info = {'type': 'Polynomial',
                'params': params,
                'data': data,
                }
        return info

    @staticmethod
    def from_json(js):
        init = {}
        A = np.array(js['data']['A'])
        init['A'] = A
        init.update(js['params'])
        trans = Polynomial(**init)
        return trans

    def fit(self, inx, iny, outx, outy, invert=False): #, exact=False):
        # to arrays
        inx = np.array(inx)
        iny = np.array(iny)
        outx = np.array(outx)
        outy = np.array(outy)

        # auto determine order from number of points
        if not self.order:
            # due to automation and high likelihood of errors, we set higher point threshold for polynomial order
            # compare to gdal: https://github.com/naturalatlas/node-gdal/blob/master/deps/libgdal/gdal/alg/gdal_crs.c#L186
            if len(inx) >= 20:
                self.order = 3
            if len(inx) >= 10:
                self.order = 2
            else:
                self.order = 1
            # update minpoints
            self.minpoints = {1:3, 2:10, 3:20}[self.order] 
        
        if self.order == 1:
            # terms
            x = inx
            y = iny
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([x,y,ones]).transpose()
            # x and y coeffs
            xcoeffs,xres,xrank,xsing = np.linalg.lstsq(u, outx, rcond=-1) 
            ycoeffs,yres,yrank,ysing = np.linalg.lstsq(u, outy, rcond=-1)
            # A matrix
            A = np.eye(3)
            # two first rows of the A matrix are equations for the x and y coordinates, respectively
            A[0,:] = xcoeffs
            A[1,:] = ycoeffs
            # get inverse transform by inverting the Matrix
            if invert:
                A = np.linalg.inv(A)
            
        elif self.order == 2:
            
            if invert:
                # fit the forward transform
                forward = self.copy()
                forward.fit(inx, iny, outx, outy)
                
                # forward predict regularly spaced sample points across the range of the inpoints
                xmin,ymin,xmax,ymax = inx.min(), iny.min(), inx.max(), iny.max()
                x = np.linspace(xmin, xmax, 100)
                y = np.linspace(ymin, ymax, 100)
                x,y = np.meshgrid(x, y)
                x,y = x.flatten(), y.flatten()
                x_pred,y_pred = forward.predict(x, y)

                # get backward transform by fitting the forward predicted sample points to the sample points
                # should be a near perfect match (~0 residuals) since these are derived from the same transform
                backward = self.copy()
                backward.fit(x_pred, y_pred, x, y)
                A = backward.A

##                import accuracy
##                x_backpred,y_backpred = backward.predict(x_pred, y_pred)
##                dists = accuracy.distances(x, y, x_backpred, y_backpred)
##                print('!!! max resid', dists.max())
##                print('!!! resid rmse', accuracy.RMSE(dists))

            else:
                # terms
                x = inx
                y = iny
                xx = x*x
                xy = x*y
                yy = y*y
                ones = np.ones(x.shape)
                # u consists of each term in equation, with each term being array if want to transform multiple
                u = np.array([xx,xy,yy,x,y,ones]).transpose()
                # find best coefficients for all equivalent points using least squares
                xcoeffs,xres,xrank,xsing = np.linalg.lstsq(u, outx, rcond=-1) 
                ycoeffs,yres,yrank,ysing = np.linalg.lstsq(u, outy, rcond=-1)
                # A matrix
                A = np.eye(6)
                # two first rows of the A matrix are equations for the x and y coordinates, respectively
                A[0,:] = xcoeffs
                A[1,:] = ycoeffs

        elif self.order == 3:
            
            if invert:
                # fit the forward transform
                forward = self.copy()
                forward.fit(inx, iny, outx, outy)
                
                # forward predict regularly spaced sample points across the range of the inpoints
                xmin,ymin,xmax,ymax = inx.min(), iny.min(), inx.max(), iny.max()
                x = np.linspace(xmin, xmax, 100)
                y = np.linspace(ymin, ymax, 100)
                x,y = np.meshgrid(x, y)
                x,y = x.flatten(), y.flatten()
                x_pred,y_pred = forward.predict(x, y)

                # get backward transform by fitting the forward predicted sample points to the sample points
                # should be a near perfect match (~0 residuals) since these are derived from the same transform
                backward = self.copy()
                backward.fit(x_pred, y_pred, x, y)
                A = backward.A

            else:
                # terms
                #X = a0 + a1x + a2y + a3xy + a4x^2 + a5y^2 + a6x^3 + a7x^2y + a8xy^2 + a9y^3
                #Y = b0 + b1x + b2y + b3xy + b4x^2 + b5y^2 + b6x^3 + b7x^2y + b8xy^2 + b9y^3
                x = inx
                y = iny
                xx = x*x
                xy = x*y
                yy = y*y
                xxx = xx*x
                xxy = xx*y
                xyy = x*yy
                yyy = yy*y
                ones = np.ones(x.shape)
                # u consists of each term in equation, with each term being array if want to transform multiple
                u = np.array([xxx,xxy,xyy,yyy, xx,xy,yy, x,y,ones]).transpose()
                # x and y coeffs
                xcoeffs,xres,xrank,xsing = np.linalg.lstsq(u, outx, rcond=-1) 
                ycoeffs,yres,yrank,ysing = np.linalg.lstsq(u, outy, rcond=-1)
                # A matrix
                A = np.eye(10)
                # two first rows of the A matrix are equations for the x and y coordinates, respectively
                A[0,:] = xcoeffs
                A[1,:] = ycoeffs

        self.A = A
        return self

    def predict(self, x, y):
        # to arrays
        x = np.array(x)
        y = np.array(y)

        # input
        u = np.array([x,y])
        
        if self.order == 1:
            # terms
            x = x
            y = y
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([x,y,ones])
            
        elif self.order == 2:
            # terms
            x = x
            y = y
            xx = x*x
            xy = x*y
            yy = y*y
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([xx,xy,yy,x,y,ones])

        elif self.order == 3:
            # terms
            x = x
            y = y
            xx = x*x
            xy = x*y
            yy = y*y
            xxx = xx*x
            xxy = xx*y
            xyy = x*yy
            yyy = yy*y
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([xxx,xxy,xyy,yyy, xx,xy,yy, x,y,ones])

        # apply the transform matrix to predict output
        predx,predy = self.A.dot(u)[:2]
        return predx,predy



class Projection(object):
    def __init__(self, fromcrs, tocrs):
        '''Map projection transform'''
        import pycrs
        self.fromcrs = pycrs.parse.from_unknown_text(fromcrs)
        self.tocrs = pycrs.parse.from_unknown_text(tocrs)
        self.minpoints = 0

    def __repr__(self):
        return u'Projection Transform(fromcrs={}, tocrs={})'.format(self.fromcrs.to_proj4(), self.tocrs.to_proj4())

    def copy(self):
        new = Projection(fromcrs=self.fromcrs, topoints=self.topoints)
        new.minpoints = self.minpoints
        return new

    def info(self):
        # TODO: rename to_json()
        params = {}
        data = {'fromcrs': self.fromcrs.to_proj4(),
                'tocrs': self.tocrs.to_proj4()}
        info = {'type': 'Projection',
                'params': params,
                'data': data,
                }
        return info

    @staticmethod
    def from_json(js):
        init = {}
        init['fromcrs'] = js['data']['fromcrs']
        init['tocrs'] = js['data']['tocrs']
        trans = Projection(**init)
        return trans

    def fit(self, *args, **kwargs):
        raise Exception('The map projection transform is an analytic transformation and does not need to be fit or estimated')

    def predict(self, x, y):
        import pyproj
        fromcrs = pyproj.Proj(self.fromcrs.to_proj4())
        tocrs = pyproj.Proj(self.tocrs.to_proj4())
        predx,predy = pyproj.transform(fromcrs,
                               tocrs,
                               x, y)
        return predx,predy



class TIN(object):
    def __init__(self):
        '''Creates a triangulated irregular network (TIN) between control points
        and does a global affine transform within each triangle'''
        self.tris = []
        self.minpoints = 3 # at least one triangle/affine

    def __repr__(self):
        return u'TIN Transform(estimated={})'.format(bool(self.tris))

    def copy(self):
        new = TIN()
        new.tris = list(self.tris)
        new.minpoints = self.minpoints
        return new

    def info(self):
        params = {}
        tri_models = [(tri,trans.info()) for tri,trans in self.tris]
        data = {'tris': tri_models}
        info = {'type': 'TIN',
                'params': params,
                'data': data,
                }
        return info

    def fit(self, inx, iny, outx, outy, invert=False):
        # to arrays
        inx = np.array(inx)
        iny = np.array(iny)
        outx = np.array(outx)
        outy = np.array(outy)

        import shapely

        inpoints = zip(inx,iny)
        inpoints = shapely.geometry.MultiPoint(inpoints)
        intris = shapely.ops.triangulate(inpoints)

        self.tris = []
        for intri in intris:
            intri_points = list(intri.exterior.coords)[:3]
            intri_x,intri_y = zip(*intri_points)
            outtri_x = [outx[inx==_x][0] for _x in intri_x]
            outtri_y = [outy[iny==_y][0] for _y in intri_y]
            outtri_points = zip(outtri_x, outtri_y)
            if invert:
                trans = Polynomial(1)
                trans.fit(intri_x, intri_y, outtri_x, outtri_y, invert=True)
                self.tris.append((outtri_points, trans))
            else:
                trans = Polynomial(1)
                trans.fit(intri_x, intri_y, outtri_x, outtri_y)
                self.tris.append((intri_points, trans))

    def predict(self, x, y):

        def point_in_tri(x1,y1,x2,y2,x3,y3,xp,yp):
            # https://www.w3resource.com/python-exercises/basic/python-basic-1-exercise-40.php
            # modified to work with numpy
            c1 = (x2-x1)*(yp-y1)-(y2-y1)*(xp-x1)
            c2 = (x3-x2)*(yp-y2)-(y3-y2)*(xp-x2)
            c3 = (x1-x3)*(yp-y3)-(y1-y3)*(xp-x3)
            intri = ((c1<=0) & (c2<=0) & (c3<=0)) | ((c1>=0) & (c2>=0) & (c3>=0))
            return intri

        x = np.array(x, np.float32)
        y = np.array(y, np.float32)
        #print 'inx',x
        #print 'iny',y

        predx = np.ones((len(x),)) * float('nan')
        predy = np.ones((len(y),)) * float('nan')
        for tri,trans in self.tris:
            #print tri
            (x1,y1),(x2,y2),(x3,y3) = tri
            intri = point_in_tri(x1,y1,x2,y2,x3,y3, x,y)
            trix = x[intri]
            triy = y[intri]
            if len(trix):
                #print len(trix)
                predtrix,predtriy = trans.predict(trix, triy)
                predx[intri] = predtrix
                predy[intri] = predtriy

        return predx, predy




    
