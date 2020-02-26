
import numpy as np

def from_json(js):
    cls = {'Polynomial':Polynomial,
           'TIN':TIN,
           }[js['type']]
    trans = cls.from_json(js)
    return trans

class Polynomial(object):
    
    def __init__(self, order=None, A=None):
        '''Polynomial transform of 1st (affine), 2nd, or 3rd order'''
        if A is not None:
            A = np.array(A)
            if A.shape == (3,3):
                order = 1
            elif A.shape == (6,6):
                order = 2
            else:
                raise ValueError('Matrix A must be shape (3,3), or (6,6); not {}'.format(A.shape))

        self.A = A
        self.order = order
        self.minpoints = {1:3, 2:6, 3:9}.get(order, 3) # minimum 3 if order not set

    def __repr__(self):
        return u'Polynomial Transform(order={}, estimated={})'.format(self.order, self.A is not None)

    def copy(self):
        new = Polynomial(order=self.order, A=self.A)
        new.minpoints = self.minpoints
        return new

    def info(self):
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

    def fit(self, inx, iny, outx, outy, invert=False):
        # to arrays
        inx = np.array(inx)
        iny = np.array(iny)
        outx = np.array(outx)
        outy = np.array(outy)

        # auto determine order from number of points
        if not self.order:
            # due to automation and high likelihood of errors, we set higher point threshold for polynomial order
            # compare to gdal: https://github.com/naturalatlas/node-gdal/blob/master/deps/libgdal/gdal/alg/gdal_crs.c#L186
            #if len(inx) >= 20:
            #    self.order = 3
            if len(inx) >= 10:
                self.order = 2
            else:
                self.order = 1
            # update minpoints
            self.minpoints = {1:3, 2:6, 3:9}[self.order] 
        
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
            # get inverse transform by switching the from and to coords (warning, not an exact inverse bc different lstsq estimation)
            if invert:
                inx,iny,outx,outy = outx,outy,inx,iny
            # terms
            x = inx
            y = iny
            xx = x*x
            xy = x*y
            yy = y*y
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([xx,xy,yy,x,y,ones]).transpose()
            # x and y coeffs
            xcoeffs,xres,xrank,xsing = np.linalg.lstsq(u, outx, rcond=-1) 
            ycoeffs,yres,yrank,ysing = np.linalg.lstsq(u, outy, rcond=-1)
            # A matrix
            A = np.eye(6)
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

        # apply the transform matrix to predict output
        predx,predy = self.A.dot(u)[:2]
        return predx,predy



class TIN:
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




    
