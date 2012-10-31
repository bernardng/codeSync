"""
Operations on the manifold of SPD matrices and mapping to a flat space.

"""
import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs

def my_stack(arrays):
    return np.concatenate([a[np.newaxis] for a in arrays])

# Bypass scipy for faster eigh (and dangerous: Nan will kill it)
my_eigh, = get_lapack_funcs(('syevr', ), np.zeros(1))

def frobenius(mat):
    """ Return the Frobenius norm
    """
    return np.sqrt((mat**2).sum())/mat.size


def sqrtm(mat):
    """ Matrix square-root, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs*np.sqrt(vals), vecs.T)


def inv_sqrtm(mat):
    """ Inverse of matrix square-root, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs/np.sqrt(vals), vecs.T)


def inv(mat):
    """ Inverse of matrix, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs/vals, vecs.T)


def expm(mat):
    """ Matrix exponential, for symetric positive definite matrices.
    """
    vals, vecs = linalg.eigh(mat)
    return np.dot(vecs*np.exp(vals), vecs.T)


def logm(mat):
    """ Matrix log, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs*np.log(vals), vecs.T)


def log_map(x, displacement, mean=False):
    """ The Riemannian log map at point 'displacement'.
        If several points are given, the mean is returned.

        See algorithm 2 of Fletcher and Joshi, Sig Proc 87 (2007) 250
    """
    #x = np.asanyarray(x)
    vals, vecs, success_flag = my_eigh(displacement)
    sqrt_vals = np.sqrt(vals)
    whitening = (vecs/sqrt_vals).T
    if len(x.shape) == 2:
        vals_y, vecs_y, success_flag = my_eigh(np.dot(np.dot(whitening, x), 
                                                whitening.T))
        sqrt_displacement = np.dot(vecs*sqrt_vals, vecs_y)
        return np.dot(sqrt_displacement*np.log(vals_y), sqrt_displacement.T)
    sqrt_displacement = vecs*sqrt_vals
    y = list()
    for this_x in x:
        vals_y, vecs_y, success_flag = my_eigh(np.dot(
                                                np.dot(whitening, this_x), 
                                                whitening.T))
        y.append(np.dot(vecs_y*np.log(vals_y), vecs_y.T))
    y = my_stack(y)
    if mean:
        y = np.mean(y, axis=0)
        return np.dot(np.dot(sqrt_displacement, y), sqrt_displacement.T)
    return my_stack([np.dot(np.dot(sqrt_displacement, this_y), 
                                sqrt_displacement.T)
                     for this_y in y])
    

def exp_map(x, displacement):
    """ The Riemannian exp map at point 'displacement'.

        See algorithm 1 of Fletcher and Joshi, Sig Proc 87 (2007) 250
    """
    vals, vecs, success_flag = my_eigh(displacement)
    sqrt_vals = np.sqrt(vals)
    whitening = (vecs/sqrt_vals).T
    vals_y, vecs_y, success_flag = my_eigh(np.dot(np.dot(whitening, x), 
                                            whitening.T))
    sqrt_displacement = np.dot(vecs*sqrt_vals, vecs_y)
    return np.dot(sqrt_displacement*np.exp(vals_y), sqrt_displacement.T)


def log_mean(population_covs, eps=1e-5):
    """ Find the Riemannien mean of the the covariances.

        See algorithm 3 of Fletcher and Joshi, Sig Proc 87 (2007) 250
    """
    step = 1.
    mean = population_covs[0]
    N = mean.size
    eps = N*eps
    direction = old_direction = log_map(population_covs, mean, mean=True)
    while frobenius(direction) > eps:
        direction = log_map(population_covs, mean, mean=True)
        mean = exp_map(step*direction, mean)
        assert np.all(np.isfinite(direction))
        if frobenius(direction) > frobenius(old_direction):
            step = .8*step
            old_direction = direction
    return mean



################################################################################
# Some tests (should be more)

def test_exp_map():
    # Check that the exponential around the identity matrix gives a
    # standard matrix exponential
    for _ in range(10):
        a = np.random.random((4, 4))
        # Make sure that we have an SPD matrix
        a = np.dot(a, a.T)
        np.testing.assert_array_almost_equal(exp_map(a, np.eye(4)), expm(a))


def test_log_map():
    # Check that the logarithm around the identity matrix gives a
    # standard matrix logarithm
    a_list = list()
    for _ in range(10):
        a = np.random.random((4, 4))
        # Make sure that we have an SPD matrix
        a = np.dot(a, a.T)
        a_list.append(a)

    a_list = np.array(a_list)
    a_logs = log_map(a_list, np.eye(4))
    for a, a_log in zip(a_list, a_logs):
        np.testing.assert_array_almost_equal(log_map(a, np.eye(4)), logm(a))
        np.testing.assert_array_almost_equal(a_log, logm(a))

    np.testing.assert_array_almost_equal(a_logs.mean(axis=0), 
                                         log_map(a_list, np.eye(4), mean=True))


def test_log_mean():
    # Test log_mean on a set of identity matrices: the mean of identities
    # should be identities
    id = np.eye(4)
    a = my_stack([id for _ in range(10)])
    np.testing.assert_array_almost_equal(log_mean(a), id)


