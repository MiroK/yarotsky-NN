from dolfin import *
import numpy as np
from scipy.linalg import eigvalsh


parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

set_log_level(80)

def hat_basis(shift, nlevels):
    '''Basis of shifted composed hat functions as Expression'''
    cpp_code = '''
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>

class MyCppExpression : public dolfin::Expression
{
public:
  MyCppExpression() : dolfin::Expression(), shift(0.5), nlevels(1) { }

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
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
};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<MyCppExpression, std::shared_ptr<MyCppExpression>, dolfin::Expression>
    (m, "Foo")
    .def(py::init<>())
    .def_readwrite("shift", &MyCppExpression::shift)
    .def_readwrite("nlevels", &MyCppExpression::nlevels)        
;
}

'''
    if isinstance(nlevels, (int, np.int64)):
        levels = np.arange(1, nlevels+1)
    else:
        levels = np.fromiter(nlevels, dtype='uintp')
        
    if isinstance(shift, (int, float)):
        shifts = shift*np.ones_like(levels)
    else:
        assert len(shift) == len(levels)

        shifts = np.fromiter(shift, dtype=float)
    
    basis = []
    for s, k in zip(shifts, levels):
        f = CompiledExpression(compile_cpp_code(cpp_code).Foo(), degree=5)        
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


def solve_poisson_hat(mesh, shift, nlevels, f, debug=False):
    '''-Delta u = f with homog DirichletBcs using nlevels of shifted basis functions'''
    basis = hat_basis(shift, nlevels)
    # Get basis foos as FEM mesh foos
    S = FunctionSpace(mesh, 'CG', 1)
    # So that we can compute their gradient
    basis = [interpolate(b, S) for b in basis]
    
    # Build approximation space
    V = VectorFunctionSpace(mesh, 'R', 0, len(basis))
    u, v = TrialFunction(V), TestFunction(V)
    
    u = sum(ui*fi for ui, fi in zip(u, basis))
    v = sum(vi*fi for vi, fi in zip(v, basis))

    a = inner(grad(u), grad(v))*dx
    m = inner(u, v)*dx
    L = inner(f, v)*dx

    A, M, b = map(assemble, (a, m, L))

    A_ = A.array()
    debug and print('\tA')
    debug and print('\tA', np.linalg.cond(A_))
    debug and print('\tA', np.linalg.norm(A_ - np.diag(np.diagonal(A_)), np.inf))

    M_ = M.array()
    debug and print('\tM')
    debug and print('\t', np.linalg.cond(M_))
    debug and print('\t', np.linalg.norm(M_ - np.diag(np.diagonal(M_)), np.inf))
    debug and print()

    debug and print('\t(A, M)')
    lmin, lmax = np.sort(np.abs(eigvalsh(A_, M_)))[[0, -1]]
    debug and print('\t', (lmin, lmax, lmax/lmin))
    debug and print()
    
    wh = Function(V)
    solve(A, wh.vector(), b)
    coefs = wh.vector().get_local()

    wh = Function(S)
    # NOTE: account for non-homog bcs outside
    for ci, ui in zip(coefs, basis):
        wh.vector().axpy(ci, ui.vector())

    return wh


def yarotsky(mesh, shift, nlevels, f=Constant(-2), u=Expression('x[0]*x[0]', degree=8)):
    '''Solve with x**2 solves -u'' = -2'''
    # As lincomb
    uh = solve_poisson_hat(mesh, shift=shift, nlevels=nlevels, f=f)

    x = interpolate(Expression('x[0]', degree=1), fh.function_space())
    uh.vector().axpy(1, x.vector())

    if not isinstance(u, (tuple, list)): u = (u, )

    np.array([errornorm(ui, uh, 'H10') for ui in u])


def yarotsky0(mesh, shift, nlevels, f=Constant(2), u=Expression('x[0]*(1-x[0])', degree=2)):
    '''Solve with x**2 solves -u'' = -2'''
    uh = solve_poisson_hat(mesh, shift=shift, nlevels=nlevels, f=f)

    if not isinstance(u, (tuple, list)): u = (u, )

    return np.array([errornorm(ui, uh, 'H10') for ui in u]), uh

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from functools import partial
    
    levels = np.arange(4, 10)
    shift = 0.5

    foo = yarotsky0  # shift 0.5 okay for x*(1-x)
    foo = yarotsky0  # shift 0.5 okay for x**2

    #u = Expression('pi*pi*sin(pi*x[0])', degree=5)
    #foo = lambda m, sh, nl: yarotsky0(m, sh, nl,
    #                                  #Expression('sin(pi*x[0])', degree=5),
    #                                  Constant(2),
    #                                  [u,
    #                                   Expression('x[0]*(1-x[0])', degree=2)])

    
    cond = []
    for nlevels in levels:
        print('Depth', nlevels)
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
        rel_error = lambda: np.min(np.abs(errors[-2]-errors[-1])/errors[-2])

        #print '\t', 0, errors[-2:], rel_error(), mesh.num_cells()
        while not (rel_error() < 5E-3 and len(errors) > 5):
            mesh = refine(mesh)
            # errors.append(cond_basis(mesh, shift, nlevels))
            e, uh = foo(mesh, shift, nlevels)
            errors.append(e)        
        print('error', e)
            #print '\t', len(errors)-2, errors[-2:], rel_error(), mesh.num_cells()

        cond.append(errors[-1])

    plt.figure()
    plt.semilogy(levels, cond, marker='o')

    u = Expression('x[0]*(1-x[0])', degree=2)
    
    x = np.linspace(0, 1, 1000)
    y = np.array(list(map(uh, x)))
    y0 = np.array(list(map(u, x)))

    # -------
                   
    e = y0 - y
    de_dx = np.diff(e)/np.diff(x)
    plt.figure()
    plt.plot(x, y0-y)
    # plt.plot(0.5*(x[1:]+x[:-1]), de_dx)

    plt.savefig('sympy_2019.png')
    plt.show()
