from dolfin import *
import numpy as np


parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)



def hat_basis(shift, nlevels):
    '''Basis of shifted composed hat functions as Expression'''
    cpp_code = '''
class MyCppExpression : public Expression
{
public:
  MyCppExpression() : Expression(), shift(0.5), nlevels(1) { }

  void eval(Array<double>& values, const Array<double>& x) const
  {
    double value = fmin(x[0]/shift, (1-x[0])/(1-shift));
    int k = 1;
    while(k < nlevels){
        value = fmin(value/shift, (1-value)/(1-shift));
        k += 1;
    }
    values[0] = value;
  }
public:
    double shift, nlevels;
};'''
    if isinstance(nlevels, int):
        levels = np.arange(1, nlevels+1)
    else:
        levels = np.fromtiter(nlevels, dtype='uintp')
        
    if isinstance(shift, (int, float)):
        shifts = shift*np.ones_like(levels)
    else:
        assert len(shift) == len(levels)

        shifts = np.fromiter(shift, dtype=float)
    
    basis = []
    for s, k in zip(shifts, levels):
        f = Expression(cpp_code, degree=10)
        f.shift = shift
        f.nlevels = k
        basis.append(f)
    return basis


def cond_basis(mesh, shift, nlevels):
    '''Properties of stiffness and mass matrices of the has basis'''
    basis = hat_basis(shift, nlevels)

    S = FunctionSpace(mesh, 'CG', 10)
    dbasis = [grad(interpolate(b, S)) for b in basis]
    
    # Now turn it into expressions
    V = VectorFunctionSpace(mesh, 'R', 0, len(basis))
    u, v = TrialFunction(V), TestFunction(V)
    
    gu = sum(ui*fi for ui, fi in zip(u, dbasis))
    gv = sum(vi*fi for vi, fi in zip(v, dbasis))

    u = sum(ui*fi for ui, fi in zip(u, basis))
    v = sum(vi*fi for vi, fi in zip(v, basis))
    
    a = inner(gu, gv)*dx
    m = inner(u, v)*dx

    A, M = assemble(a), assemble(m)

    Amin, Amax = np.sort(np.abs(np.linalg.eigvalsh(A.array())))[[0, -1]]
    Acond = Amax/Amin

    Mmin, Mmax = np.sort(np.abs(np.linalg.eigvalsh(M.array())))[[0, -1]]
    Mcond = Mmax/Mmin

    return {'A': (Amin, Amax, Acond),
            'M': (Mmin, Mmax, Mcond)}


def solve_poisson_fem(mesh, f):
    '''-Delta u = f with homog DirichletBcs using P1'''
    V = FunctionSpace(mesh, 'CG', 1)
    u, v = TrialFunction(V), TestFunction(V)
    
    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx
    bcs = DirichletBC(V, Constant(0), 'on_boundary')
    
    A, b = assemble_system(a, L, bcs)
    wh = Function(V)
    solve(A, wh.vector(), b)
    
    return wh


def solve_poisson_hat(mesh, shift, nlevels, f):
    '''-Delta u = f with homog DirichletBcs using nlevels of shifted basis functions'''
    basis = hat_basis(shift, nlevels)
    # Get basis foos as FEM mesh foos
    S = FunctionSpace(mesh, 'CG', 10)
    # So that we can compute their gradient
    dbasis = [grad(interpolate(b, S)) for b in basis]
    
    # Build approximation space
    V = VectorFunctionSpace(mesh, 'R', 0, len(basis))
    u, v = TrialFunction(V), TestFunction(V)
    
    gu = sum(ui*fi for ui, fi in zip(u, dbasis))
    gv = sum(vi*fi for vi, fi in zip(v, dbasis))
    v = sum(vi*fi for vi, fi in zip(v, basis))

    a = inner(gu, gv)*dx
    L = inner(f, v)*dx

    A, b = map(assemble, (a, L))

    A_ = A.array()
    print np.linalg.cond(A_), '<<<<<<<<<<<'
    print np.linalg.norm(A_ - np.diag(np.diagonal(A_)), np.inf), '>>>>>>>>>>'
    
    wh = Function(V)
    solve(A, wh.vector(), b)
    # NOTE: account for non-homog bcs outside
    wh = sum(whi*fi for whi, fi in zip(wh, basis))

    return wh


def yarotsky(mesh, shift, nlevels, f=Constant(-2), u=Expression('x[0]*x[0]', degree=8)):
    '''Solve with x**2 solves -u'' = -2'''
    f = Constant(-2)

    # As lincomb
    fh = solve_poisson_hat(mesh, shift=shift, nlevels=nlevels, f=f)
    # We are not after the error
    Vh = FunctionSpace(mesh, 'CG', 3)
    # Accounts for homog bcs in solve
    x, = SpatialCoordinate(mesh)
    uh = project(x + uh, Vh)
    
    return errornorm(u, uh, 'H10'), fh


def yarotsky0(mesh, shift, nlevels, f=Constant(2), u=Expression('x[0]*(1-x[0])', degree=8)):
    '''Solve with x**2 solves -u'' = -2'''
    f = Constant(2)

    # As lincomb
    uh = solve_poisson_hat(mesh, shift=shift, nlevels=nlevels, f=f)
    Vh = FunctionSpace(mesh, 'CG', 3)
    uh = project(uh, Vh)
    
    return errornorm(u, uh, 'H10'), uh

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from functools import partial
    
    levels = np.arange(4, 10)
    shift = 0.5

    u = Expression('x[0]*x[0]*(1-x[0])*(1-x[0])', degree=8)

    foo = yarotsky0  # shift 0.5 okay for x*(1-x)
    foo = yarotsky0  # shift 0.5 okay for x**2
    foo = partial(yarotsky0,
                  f=Expression(' x[0]*x[0]*(2*x[0] - 2) + 2*x[0]*(-x[0] + 1)*(-x[0] + 1)', degree=4),
                  u=u)
    
    cond = []
    for nlevels in levels:
        print 'Depth', nlevels
        # Run over meshes until independence
        errors = []
        
        mesh = UnitIntervalMesh(1024)
        e, uh = foo(mesh, shift, nlevels)
        errors.append(e)
        #errors.append(cond_basis(mesh, shift, nlevels))
        
        mesh = refine(mesh)
        e, uh = foo(mesh, shift, nlevels)
        errors.append(e)        
        #errors.append(cond_basis(mesh, shift, nlevels))
        rel_error = lambda: np.abs(errors[-2]-errors[-1])/errors[-2]

        print '\t', 0, errors[-2:], rel_error(), mesh.num_cells()
        while not (rel_error() < 5E-3 and len(errors) > 5):
            mesh = refine(mesh)
            # errors.append(cond_basis(mesh, shift, nlevels))
            e, uh = foo(mesh, shift, nlevels)
            errors.append(e)        
            
            print '\t', len(errors)-2, errors[-2:], rel_error(), mesh.num_cells()

        cond.append(errors[-1])

    plt.figure()
    plt.plot(levels, cond)

    #x = np.linspace(0, 1, 1000)
    #y = [uh(xi) for xi in x]
    #y0 = [uh(xi) for xi in x]
    
    #plt.figure()
    #plt.plot(x, y)
    #plt.plot(x, y0)
    #plt.show()
