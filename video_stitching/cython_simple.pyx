# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np
from time import time
import sys

from cython.parallel import prange, parallel, threadid

ctypedef np.float64_t dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
def coor_load(np.ndarray[dtype_t, ndim=3]  img_L  ,
              np.ndarray[dtype_t, ndim=3]  img_R  , 
              np.ndarray[np.int64_t, ndim=3] coor ,
              int height ,
              int width):

    cdef int h = height
    cdef int w = width
    cdef Py_ssize_t i, j,x,y
    cdef double s = time()

    for i in prange(h , nogil=True):
        for j in prange(w , nogil=True):
            y = coor[i,j,0]
            x = coor[i,j,1]
            if (x < 0  or y < 0 ):
                continue
            img_L[i,j] = img_R[x,y]
    print("Cython runtime" , time()-s)
    return img_L

"""
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef matmul(numpy.ndarray[dtype_t,ndim=2] A,
             numpy.ndarray[dtype_t,ndim=1] B):
    cdef numpy.ndarray[dtype_t,ndim=1] out = numpy.zeros((A.shape[0]))
    cdef Py_ssize_t i,j,k
    cdef dtype_t s
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            s = 0
            for k in range(A.shape[1]):
                s += A[i,k] * B[j]
            out[i] = s
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef remap(numpy.ndarray[double, ndim=2] H ,numpy.ndarray[double, ndim=3] img_left , numpy.ndarray[double, ndim=3] img_right, int width , int height,int wr , int hr):
    a_s = time()
    cdef numpy.ndarray[dtype_t] coor = numpy.array([0, 0, 1] , dtype = 'float64');
    cdef numpy.ndarray res = numpy.empty((width,height,3));
    
    #res = numpy.array([])
    #res_list = list()

    cdef int i
    cdef int j

    for i in range(width):
        for j in range(height):
            #print(width , height)
            
            coor[0]  = j
            coor[1]  = i
            
            #img_right_coor_a =  numpy.matmul(H , coor) #H @ coor    2.1457672119140625e-06

            img_right_coor  = H @ coor 
            
            img_right_coor /= img_right_coor[2]   
            
            y, x = int(numpy.round(img_right_coor[0])), int(numpy.round(img_right_coor[1])) 
              
            if (x < 0 or x >= hr or y < 0 or y >= wr):
                continue
            
            img_left[i, j] = img_right[x, y]
            
    print('Cython runtime :' , time() - a_s)
    return img_left    
"""