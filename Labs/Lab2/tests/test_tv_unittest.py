import unittest
import numpy as np

def gradient2D(X):
    nX = np.copy(X)

    if len(np.shape(nX)) > 2:
        raise ValueError("Dimension superior to 2.")

    XDh = np.diff(nX)
    XDh = np.c_[XDh, np.zeros(len(nX))]

    DvX = np.diff(nX.T)
    DvX = np.c_[DvX, np.zeros(len(nX.T))].T

    D = (XDh, DvX)

    return D

def tv(X):
    D = gradient2D(X)
    XDh = D[0]
    DvX = D[1]

    TV = 0

    for i in range(len(X)) :
        for j in range(len(X[0])) :
            TV += np.sqrt(XDh[i][j]**2 + DvX[i][j]**2)

    return TV

class TestClass(unittest.TestCase):
    def test_dim(self):
        M = [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]
        self.assertRaises(ValueError, tv, M)

    def test_squared(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        TV = tv(M)

        assert (TV == 4*(2 + np.sqrt(10)))

    def test_not_squared(self):
        M = [[1, 2, 3, 4], [7, 7, 7, 7]]

        TV = tv(M)

        assert (TV == 3 + np.sqrt(17) + np.sqrt(37) + np.sqrt(26))

if __name__ == "__main__":
    unittest.main()
