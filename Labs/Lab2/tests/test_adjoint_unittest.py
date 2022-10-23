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


def trace1D(X, Y):
    sum = 0
    nX = np.array(X)
    nY = np.array(Y)
    for i in range(len(nX.T)):
        for j in range(len(nY)):
            sum += nX.T[i][j] * Y[j][i]

    return sum


def trace2D(X1, X2, Y1, Y2):
    return trace1D(X1, Y1) + trace1D(X2, Y2)


def gradient2D_adjoint(Yh, Yv):
    nYh = np.copy(Yh)  # Yh copy
    nYv = np.copy(Yv)  # Yv copy

    YhDh = []
    DvYv = []

    assert nYh.shape == nYv.shape, "Yh and Yv dimensions are not equal!"
    assert (
        nYh.shape[0] != 1 or nYh.shape[1] != 1
    ), "Unable to use the formulas with scalars. Ex: X = 1, Yh = 1, Yv = 2 -> 0 = -3!"
    assert (
        nYh.shape[0] != 2 or nYh.shape[1] != 1
    ), "Unable to use the formulas with 2x1 matrix!"
    assert (
        nYh.shape[0] != 1 or nYh.shape[1] != 2
    ), "Unable to use the formulas with 1x2 matrix!"

    if (
        nYh.shape[0] == 1 & nYh.shape[1] == 1
    ):  # 1x1 matrix case, but we will never use it because of the 2.6 formulas that are only working for 2x2 matrices and above

        YhDh = np.array([-nYh[0]])
        DvYv = np.array([-nYv[0]])

    else:

        nYh1 = nYh[:, 0]  # We save Yh's first column
        nYhN = nYh[:, len(nYh[0]) - 2]  # We save Yh's before last column

        YhDh = np.diff(nYh)  # We compute the intermediary columns
        if len(YhDh[0]) - 1 > 0:  # We make sure they exist
            YhDh = np.c_[
                -nYh1, -YhDh[:, : len(YhDh[0]) - 1]
            ]  # If yes, we add them after the first column
        else:
            YhDh = np.array([-nYh1]).T  # Else, we only keep the first column

        YhDh = np.c_[
            YhDh, nYhN
        ]  # We add the before last column to obtain the final result

        nYv1 = nYv[0, :]  # We save Yv's first row
        nYvN = nYv[len(nYv) - 2, :]  # We save Yv's before last column

        DvYv = (np.diff(nYv.T)).T  # We compute the intermediary rows
        if len(DvYv) - 1 > 0:  # We make sure they exist
            DvYv = np.c_[
                -nYv1, -DvYv[: len(DvYv) - 1, :].T
            ].T  # If yes, we add them after the first row
        else:
            DvYv = np.array([-nYv1])  # Else, we only keep the first row

        DvYv = np.c_[
            DvYv.T, nYvN
        ].T  # We add the before last row to obtain the final result

    return YhDh + DvYv  # We add the two resulting matrices


class TestClass(unittest.TestCase):
    def test_shape(self):
        M1 = [[1, 1, 1, 0], [0, 0, 0, 0]]

        M2 = [[6, 5, 4, 3], [0, 0, 0, 0]]

        D = gradient2D_adjoint(M1, M2)

        assert np.array_equal(np.shape(D), [2, 4])

    def test_is_adjoint_squared_smallsize(self):
        np.random.seed(0)
        M = np.random.randint(10, size=(2, 2)) + 1j * np.random.randint(10, size=(2, 2))

        D = gradient(M)

        M1 = np.random.randint(10, size=(2, 2)) + 1j * np.random.randint(
            10, size=(2, 2)
        )

        M2 = np.random.randint(10, size=(2, 2)) + 1j * np.random.randint(
            10, size=(2, 2)
        )

        G = gradient2D_adjoint(M1, M2)

        assert trace2D(D[0], D[1], M1, M2) == trace1D(M, G)

    def test_is_adjoint_squared_bigsize(self):
        np.random.seed(0)
        M = np.random.randint(10, size=(4, 4)) + 1j * np.random.randint(10, size=(4, 4))

        D = gradient(M)

        M1 = np.random.randint(10, size=(4, 4)) + 1j * np.random.randint(
            10, size=(4, 4)
        )

        M2 = np.random.randint(10, size=(4, 4)) + 1j * np.random.randint(
            10, size=(4, 4)
        )

        G = gradient2D_adjoint(M1, M2)

        assert trace2D(D[0], D[1], M1, M2) == trace1D(M, G)

    def test_is_adjoint_notsquared_smallsize(self):
        np.random.seed(0)
        M = np.random.randint(10, size=(2, 3)) + 1j * np.random.randint(10, size=(2, 1))

        D = gradient(M)

        M1 = np.random.randint(10, size=(2, 3)) + 1j * np.random.randint(
            10, size=(2, 1)
        )

        M2 = np.random.randint(10, size=(2, 3)) + 1j * np.random.randint(
            10, size=(2, 1)
        )

        G = gradient2D_adjoint(M1, M2)
        assert trace2D(D[0], D[1], M1, M2) == trace1D(M, G)

    def test_is_adjoint_notsquared_bigsize(self):
        np.random.seed(0)
        M = np.random.randint(10, size=(5, 4)) + 1j * np.random.randint(10, size=(5, 4))

        D = gradient(M)

        M1 = np.random.randint(10, size=(5, 4)) + 1j * np.random.randint(
            10, size=(5, 4)
        )

        M2 = np.random.randint(10, size=(5, 4)) + 1j * np.random.randint(
            10, size=(5, 4)
        )

        G = gradient2D_adjoint(M1, M2)

    def test_is_adjoint_1x1(self):

        M = [[1]]
        D = gradient(M)
        M1 = [[5]]
        M2 = [[3]]

        with self.assertRaises(Exception) as context:
            gradient2D_adjoint(M1, M2)

        self.assertTrue(
            "Unable to use the formulas with scalars. Ex: X = 1, Yh = 1, Yv = 2 -> 0 = -3!"
            in str(context.exception)
        )

    def test_is_adjoint_2x1(self):

        M1 = [[3], [4]]
        M2 = [[5], [6]]

        with self.assertRaises(Exception) as context:
            gradient2D_adjoint(M1, M2)

        self.assertTrue(
            "Unable to use the formulas with 2x1 matrix!" in str(context.exception)
        )

    def test_is_adjoint_1x2(self):

        M3 = [[3, 4]]
        M4 = [[5, 6]]

        with self.assertRaises(Exception) as context:
            gradient2D_adjoint(M3, M4)

        self.assertTrue(
            "Unable to use the formulas with 1x2 matrix!" in str(context.exception)
        )

    def test_different_dimensions(self):
        np.random.seed(0)
        M1 = np.random.randint(10, size=(5, 4))
        M2 = np.random.randint(10, size=(3, 7))

        with self.assertRaises(Exception) as context:
            gradient2D_adjoint(M1, M2)

        self.assertTrue("Yh and Yv dimensions are not equal!" in str(context.exception))

    def test_grad2D_adjoint_dim(self):
        np.random.seed(0)
        M1 = np.random.randint(10, size=(5, 4)) + 1j * np.random.randint(
            10, size=(5, 4)
        )

        M2 = np.random.randint(10, size=(5, 4)) + 1j * np.random.randint(
            10, size=(5, 4)
        )

        G = gradient2D_adjoint(M1, M2)
        assert np.array_equal(np.shape(G), [5, 4])


if __name__ == "__main__":
    unittest.main()
