import unittest
import numpy as np


def gradient(X):
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


class TestClass(unittest.TestCase):
    def test_shape(self):
        M = [[1, 2, 3, 4], [7, 7, 7, 7]]

        D = gradient(M)

        assert np.array_equal(np.shape(D[0]), [2, 4])
        assert np.array_equal(np.shape(D[1]), [2, 4])

    def test_dim(self):
        M = [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]
        with self.assertRaises(Exception) as context:
            gradient(M)

        self.assertTrue(
            "Matrix dimension is above 2 !" in str(context.exception)
        )

    def test_squared(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        D = gradient(M)

        assert np.array_equal(D[0], [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        assert np.array_equal(D[1], [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [0.0, 0.0, 0.0]])

    def test_not_squared(self):
        M = [[1, 2, 3, 4], [7, 7, 7, 7]]

        D = gradient(M)

        assert np.array_equal(D[0], [[1, 1, 1, 0], [0, 0, 0, 0]])
        assert np.array_equal(D[1], [[6, 5, 4, 3], [0, 0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
