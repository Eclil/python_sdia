import unittest
import numpy as np


def gradient2D(X):
    nX = np.copy(X)
    assert len(np.shape(nX)) <= 2, "Matrix dimension is above 2 !"

    XDh = np.diff(nX)
    DvX = np.diff(nX.T)

    if len(XDh) != 0:
        XDh = np.c_[XDh, np.zeros(len(nX))]
        DvX = np.c_[DvX, np.zeros(len(nX.T))].T
    else:  # 1x1 matrix case
        XDh = [0]
        DvX = [0]

    D = (XDh, DvX)

    return D


def tv(X):
    D = gradient2D(X)
    XDh = D[0]
    DvX = D[1]

    TV = 0

    for i in range(len(X)):
        for j in range(len(X[0])):
            TV += np.sqrt(XDh[i][j] ** 2 + DvX[i][j] ** 2)

    return TV


class TestClass(unittest.TestCase):
    def test_dim(self):
        M = [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]
        with self.assertRaises(Exception) as context:
            gradient2D(M)

        self.assertTrue("Matrix dimension is above 2 !" in str(context.exception))

    def test_squared(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        TV = tv(M)

        assert TV == 4 * (2 + np.sqrt(10))

    def test_not_squared(self):
        M = [[1, 2, 3, 4], [7, 7, 7, 7]]

        TV = tv(M)

        assert TV == 3 + np.sqrt(17) + np.sqrt(37) + np.sqrt(26)


if __name__ == "__main__":
    unittest.main()
