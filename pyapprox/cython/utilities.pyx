import numpy as np
cimport numpy as np
import cython
cimport cython

fused_type = cython.fused_type(cython.numeric, np.float64_t)

ctypedef np.int_t int_t
ctypedef np.int64_t int64_t

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef sub2ind_pyx(int [:] sizes, int [:] multi_index):
    """
    Map a d-dimensional index to the scalar index of the equivalent flat
    1D array

    Example:
    \f[
    \begin{bmatrix}
    1,1 & 1,2 & 1,3\\
    2,1 & 2,2 & 2,3\\
    3,1 & 3,2 & 3,3
    \end{bmatrix}
    \rightarrow
    \begin{bmatrix}
    1 & 4 & 7\\
    2 & 5 & 8\\
    3 & 6 & 9
    \end{bmatrix}
    \f]

    Parameters
    ----------
    sizes : integer 
        The number of elems in each dimension. For a 2D index
        sizes = [numRows, numCols]

    multi_index : np.ndarray (len(sizes))
       The d-dimensional index

    Return
    ------
    scalar_index : integer 
        The scalar index
    """

    cdef int ii
    cdef int num_sets = len(sizes)
    cdef int scalar_index = 0; 
    cdef int shift = 1
    for ii in range(num_sets):
        scalar_index += shift * multi_index[ii]
        shift *= sizes[ii]
    return scalar_index

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef ind2sub_pyx(int [:] sizes, int scalar_index, int num_elems, int [:] multi_index_view):
    """
    Map a scalar index of a flat 1D array to the equivalent d-dimensional index

    Example:
    \f[
    \begin{bmatrix}
    1 & 4 & 7\\
    2 & 5 & 8\\
    3 & 6 & 9
    \end{bmatrix}
    \rightarrow
    \begin{bmatrix}
    1,1 & 1,2 & 1,3\\
    2,1 & 2,2 & 2,3\\
    3,1 & 3,2 & 3,3
    \end{bmatrix}
    \f]
    
    Parameters
    ----------
    sizes : integer 
        The number of elems in each dimension. For a 2D index
        sizes = [numRows, numCols]

    scalar_index : integer 
        The scalar index
    
    num_elems : integer
        The total number of elements in the d-dimensional matrix

    Return
    ------
    multi_index : np.ndarray (len(sizes))
       The d-dimensional index
    """
    cdef int ii
    cdef int denom = num_elems
    cdef int num_sets = len(sizes)
    #multi_index = np.empty((num_sets),dtype=np.int32)
    #cdef int [:] multi_index_view = multi_index
    for ii in range(num_sets-1,-1,-1):
        denom /= sizes[ii]
        multi_index_view[ii] = scalar_index / denom;
        scalar_index = scalar_index % denom;
    #return multi_index

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def cartesian_product_pyx(input_sets, cython.numeric element, int elem_size=1):
    """
    Compute the cartesian product of an arbitray number of sets.

    The sets can consist of numbers or themselves be lists or vectors. All 
    the lists or vectors of a given set must have the same number of entries
    (elem_size). However each set can have a different number of sclars, lists,
    or vectors.

    Parameters
    ----------
    input_sets 
        The sets to be used in the cartesian product.

    elem_size : integer 
        The size of the vectors within each set.

    element : scalar
        An element from one of the inputs sets. This is necessary so cython
	can choose correct extension based upon the type of this element.

    Return
    ------
    result : np.ndarray (num_sets*elem_size, num_elems)
        The cartesian product. num_elems = np.prod(sizes)/elem_size,
        where sizes[ii] = len(input_sets[ii]), ii=0,..,num_sets-1.
        result.dtype will be set to the first entry of the first input_set
    """

    cdef int ii,jj,kk
    cdef int num_elems = 1;
    cdef int num_sets = len(input_sets)
    cdef int [:] sizes = np.empty((num_sets),dtype=np.int32)
    cdef cython.numeric [:] input_set
    
    for ii in range(num_sets):
        sizes[ii] = input_sets[ii].shape[0]/elem_size
        num_elems *= sizes[ii]

    #cdef int [:] multi_index
    cdef int [:] multi_index = np.empty((num_sets),dtype=np.int32)
    result = np.empty(
        (num_sets*elem_size, num_elems), dtype=type(input_sets[0][0]))
    cdef cython.numeric [:,:] result_view = result
    for ii in range(num_elems):
        #multi_index = ind2sub_pyx(sizes, ii, num_elems)
        ind2sub_pyx(sizes, ii, num_elems, multi_index)
        for jj in range(num_sets):
            input_set = input_sets[jj]
            for kk in range(elem_size):
                result_view[jj*elem_size+kk,ii]=\
                  input_set[multi_index[jj]*elem_size+kk];
    return result

# if get bug. Maybe that cython and numpy install are inconsistent.
# try reinstalling conda environment
# def cartesian_product_pyx(input_sets, mixed_type element, int elem_size=1):
# TypeError: No matching signature found

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
cpdef outer_product_pyx(input_sets, fused_type element):
    """
    Construct the outer product of an arbitary number of sets.
 
    Example:
    \f[ \{1,2\}\times\{3,4\}=\{1\times3, 2\times3, 1\times4, 2\times4\} =
    \{3, 6, 4, 8\} \f]

    Parameters
    ----------
    input_sets  
        The sets to be used in the outer product

    element : scalar
        An element from one of the inputs sets. This is necessary so cython
	can choose correct extension based upon the type of this element.


    Return
    ------
    result : np.ndarray(np.prod(sizes))
       The outer product of the sets.
       result.dtype will be set to the first entry of the first input_set
    """
    cdef int ii
    cdef int num_elems = 1
    cdef int num_sets = len(input_sets)
    cdef fused_type [:] input_set

    cdef int [:] sizes = np.empty((num_sets),dtype=np.int32)
    for ii in range(num_sets):
        sizes[ii] = input_sets[ii].shape[0]
        num_elems *= sizes[ii]

    #cdef int [:] multi_index
    cdef int [:] multi_index = np.empty((num_sets),dtype=np.int32)
    result = np.empty((num_elems), dtype=input_sets[0].dtype)
    cdef fused_type [:] result_view = result
    for ii in range(num_elems):
        result_view[ii] = 1
        #multi_index = ind2sub_pyx(sizes, ii, num_elems)
        ind2sub_pyx(sizes, ii, num_elems, multi_index)
        for jj in range(num_sets):
            input_set = input_sets[jj]	
	    # Following fails for complex numbers
            # result_view[ii] *= input_set[multi_index[jj]]
	    # use next line instead
            result_view[ii] = result_view[ii]*input_set[multi_index[jj]]
          
    return result

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def halton_sequence_pyx(int64_t[:] primes, int_t index1, int_t index2):
    cdef:
        Py_ssize_t ii, kk
        int64_t summand
        Py_ssize_t num_vars = primes.shape[0]
        int64_t[:] ff = np.empty((num_vars), dtype=np.int64)
        double[:] prime_inv = np.empty((num_vars), dtype=np.double)
    
    sequence = np.zeros((num_vars, index2-index1), dtype=np.double)
    cdef double [:,:] seq_view = sequence
    kk=0
    for ii in range(index1, index2):
        ff[:] = ii
        for jj in range(num_vars):
            prime_inv[jj] = 1./primes[jj]
        summand = ii*num_vars
        while summand>0:
            summand = 0
            for jj in range(num_vars):
                seq_view[jj,kk] += (ff[jj]%primes[jj])*prime_inv[jj]
                prime_inv[jj] /= primes[jj]
                ff[jj] = ff[jj] // primes[jj]
                summand += ff[jj]
        kk+=1
    return sequence


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef swap_rows(np.ndarray matrix, Py_ssize_t ii, Py_ssize_t jj):
    matrix[ii], matrix[jj] = matrix[jj], matrix[ii]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef create_new_columns(np.ndarray[double, ndim=2] LU_factor, np.ndarray[double, ndim=2] new_cols, np.ndarray[int_t] raw_pivots):
    cdef:
        Py_ssize_t num_pivots = raw_pivots.shape[0]
        Py_ssize_t next_idx, it, ii, jj, kk
        int_t pivot

    for it, pivot in enumerate(raw_pivots):
        swap_rows(new_cols, it, pivot)

        # update LU_factor
        # recover state of col vector from permuted LU factor
        # Let  (jj,kk) represent iteration and pivot pairs
        # then if lu factorization produced sequence of pairs
        # (0,4),(1,2),(2,4) then LU_factor[:,0] here will be col_vector
        # in LU algorithm with the second and third permutations
        # so undo these permutations in reverse order
        next_idx = it+1
        col_vector = np.array(LU_factor[next_idx:, it])
        for ii in range(num_pivots-it-1):
            # (it+1) necessary in two lines below because only dealing
            # with compressed col vector which starts at row it in LU_factor
            jj = raw_pivots[num_pivots-1-ii]-next_idx
            kk = num_pivots-ii-1-next_idx

            swap_rows(col_vector, jj, kk)

        new_cols[next_idx:, :] -= np.outer(col_vector, new_cols[it, :])

    return new_cols
