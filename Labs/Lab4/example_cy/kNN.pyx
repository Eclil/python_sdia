import numpy as np

def kNN(int[:] class_train, double[:,:] train_set, double[:,:] test_set, int k):

    cdef Py_ssize_t n_xtrain = train_set.shape[0]
    cdef Py_ssize_t n_xtest = test_set.shape[0]

    distance = np.zeros(n_xtrain, dtype = float)
    cdef double[:] distance_c= distance

    class_neigh = np.zeros(k, dtype = np.intp)
    cdef Py_ssize_t[:] class_neigh_c = class_neigh

    k_nearest = np.zeros(k, dtype = np.intp)
    cdef Py_ssize_t[:] k_nearest_c = k_nearest

    count = np.zeros(3, dtype = np.intp)
    cdef Py_ssize_t[:] count_c = count

    class_pred = np.zeros(n_xtest, dtype = int)
    cdef int[:] class_pred_c= class_pred

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t index

    cdef Py_ssize_t predicted_class
    cdef int occurrences

    cdef Py_ssize_t i_count

    cdef double train_x
    cdef double train_y
    cdef double test_x
    cdef double test_y

    for i in range(n_xtest):

        """ Distances """

        test_x = test_set[i,0]
        test_y = test_set[i,1]

        for j in range(n_xtrain):
            train_x = train_set[j,0]
            train_y = train_set[j,1]
            distance_c[j] = (train_x - test_x)**2 + (train_y - test_y)**2

        """ Argsort """

        for index in range(k):
            k_nearest_c = np.argsort(distance_c)[:k]
            class_neigh_c[index] = class_train[k_nearest_c[index]]

        """ Count """

        for i_count in range(k):
            count_c[class_neigh_c[i_count]] += 1

        predicted_class = 0
        occurrences = 0

        for i_count in range(3):
            if count_c[i_count] > occurrences:
                predicted_class = i_count
                occurrences = count_c[predicted_class]
            count_c[i_count] = 0

        """ Prediction results """

        class_pred_c[i] = predicted_class

    return class_pred
