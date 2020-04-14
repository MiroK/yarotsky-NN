from dolfin import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)


def hat(x, shift=0.5):
    '''Piecewise linear f in H^1_0 with max at shift'''
    # NOTE: only use this on (0, 1) for it is wrong outside of it
    shift = Constant(shift)
    return conditional(x < shift,
                       x/shift,
                       (1-x)/(1-shift))


def hat_basis(x, shift, nlevels):
    '''Basis of hat, hat o hat, hat o hat o ... of shifted'''    
    f = hat(x, shift)
    
    basis = [f]
    while nlevels > 1:
        nlevels -= 1
        basis.append(hat(basis[-1], shift))

    return basis


def solve_H10(mesh, shift, nlevels, f):
    '''-Delta u = f with homog DirichletBCS using nlevels of shifted basis functions'''
    x, = SpatialCoordinate(mesh)
    basis = hat_basis(x, shift, nlevels)

    V = VectorFunctionSpace(mesh, 'R', 0, len(basis))
    u, v = TrialFunction(V), TestFunction(V)
    
    u = sum(ui*fi for ui, fi in zip(u, basis))
    v = sum(vi*fi for vi, fi in zip(v, basis))

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    A, b = map(assemble, (a, L))

    wh = Function(V)
    solve(A, wh.vector(), b)

    wh = sum(whi*fi for whi, fi in zip(wh, basis))

    return wh


def yarotsky(mesh, shift, nlevels):
    '''Solve with x**2'''
    f = Constant(-2)

    # As lincomb
    fh = solve_H10(mesh, shift=Constant(shift), nlevels=nlevels, f=f)
    # We are not after the error
    Vh = FunctionSpace(mesh, 'CG', 3)
    # Accounts for homog bcs in solve
    x, = SpatialCoordinate(mesh)
    fh = project(x + fh, Vh)
    f = Expression('x[0]*x[0]', degree=8)  # Just to make errornorm shut up

    return errornorm(f, fh, 'H10')
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt


    shift = 0.5
    for nlevels in (4, 8, 16):#8, 16):
        print nlevels
        # Run over meshes until independence
        errors = []
        
        mesh = UnitIntervalMesh(1024)
        errors.append(yarotsky(mesh, shift=shift, nlevels=nlevels))
        
        mesh = refine(mesh)
        errors.append(yarotsky(mesh, shift=shift, nlevels=nlevels))

        rel_error = lambda: abs(errors[-2]-errors[-1])/errors[-2]

        print '\t', errors[-2:], rel_error(), mesh.num_cells()
        while not rel_error() < 1E-3:
            mesh = refine(mesh)
            errors.append(yarotsky(mesh, shift=shift, nlevels=nlevels))
            print '\t', errors[-2:], rel_error(), mesh.num_cells()
            
    # NOTE: we will be solving with homog bcs

    # x = np.linspace(0, 1, 10000)
    # y = [fh(xi) for xi in x]
    # y0 = [f(xi) for xi in x]
    # plt.figure()
    # plt.plot(x, y)
    # plt.plot(x, y0)    
    # plt.show()

    
