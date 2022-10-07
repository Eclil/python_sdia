import unittest
import numpy as np


def gradient(X):
    nX = np.copy(X)

    if len(np.shape(nX)) > 2:
        raise ValueError("Dimension superior to 2.")

    XDh = np.diff(nX)
    XDh = np.c_[XDh, np.zeros(len(nX))]

    DvX = np.diff(nX.T)
    DvX = np.c_[DvX, np.zeros(len(nX.T))].T

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
    nYh = np.copy(Yh)  # Copie de Yh
    nYv = np.copy(Yv)  # Copie de Yv

    nYh1 = nYh[:, 0]  # On sauvegarde la première colonne de Yh
    nYhN = nYh[:, len(nYh) - 1]  # On sauvegarde l'avant-dernière colonne de Yh
    YhDh = np.diff(nYh)  # On calcule les colonnes intermédiaires
    if len(YhDh[0]) - 1 > 0:  # On vérifie qu'il y en a
        YhDh = np.c_[
            -nYh1, -YhDh[:, : len(YhDh[0]) - 1]
        ]  # Si oui, on les concatène à la première colonne
    else:
        YhDh = np.array([-nYh1]).T  # Sinon, on ne garde que la première colonne

    YhDh = np.c_[
        YhDh, nYhN
    ]  # On ajoute l'avant-dernière colonne pour obtenir le résultat

    nYv1 = nYv[0, :]  # On sauvegarde la première ligne de Yv
    nYvN = nYv[len(nYh) - 2, :]  # On sauvegarde l'avant-dernière ligne de Yv
    DvYv = (np.diff(nYv.T)).T  # On calcule les lignes intermédiaires
    if len(DvYv) - 1 > 0:  # On vérifie qu'il y en a
        DvYv = np.c_[
            -nYv1, -DvYv[: len(DvYv) - 1, :].T
        ].T  # Si Oui, on les concatène à la première ligne
    else:
        DvYv = np.array([-nYv1])  # Sinon on ne garde que la première ligne

    DvYv = np.c_[
        DvYv.T, nYvN
    ].T  # On ajoute l'avant-dernière ligne pour obtenir le résultat

    return YhDh + DvYv  # On somme nos deux matrices résultat


class TestClass(unittest.TestCase):
    def test_shape(self):
        M1 = [[1, 1, 1, 0], [0, 0, 0, 0]]

        M2 = [[6, 5, 4, 3], [0, 0, 0, 0]]

        D = gradient2D_adjoint(M1, M2)

        assert np.array_equal(np.shape(D), [2, 4])

    def test_is_adjoint_squared(self):
        M = [[1, 2, 3, 4], [7, 7, 7, 7]]

        D = gradient(M)

        M1 = [[1, 1, 1, 0], [0, 0, 0, 0]]

        M2 = [[6, 5, 4, 3], [0, 0, 0, 0]]

        G = gradient2D_adjoint(M1, M2)

        assert trace2D(D[0], D[1], M1, M2) == trace1D(M, G)


if __name__ == "__main__":
    unittest.main()
