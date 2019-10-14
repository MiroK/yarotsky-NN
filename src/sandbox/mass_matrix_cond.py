import dolfin as df
from scipy.linalg import eigvalsh
import numpy as np


def condM(n, precond=lambda M: np.eye(M.shape[0])):
    mesh = df.UnitIntervalMesh(n)
    V = df.FunctionSpace(mesh, 'CG', 1)

    u, v = df.TrialFunction(V), df.TestFunction(V)
    M = df.assemble(df.inner(u, v)*df.dx).array()
    B = precond(M)

    eigw = eigvalsh(M, B)
    lmin, lmax = np.sort(np.abs(eigw))[[0, -1]]

    return lmin, lmax, lmax/lmin

eye = lambda n: condM(n)

diagonal = lambda M: np.diag(np.diagonal(M))
diag = lambda n, precond=diagonal: condM(n, precond)

lumped_diagonal = lambda M: np.diag(np.sum(M, axis=1))
lumped = lambda n, precond=lumped_diagonal: condM(n, precond)

preconds = [eye, diag, lumped]
for n in (8, 16, 32, 64, 128, 256):
    print [f(n) for f in preconds]