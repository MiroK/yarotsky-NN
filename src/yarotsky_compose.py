# Yarotsky on [-1, 1]

import numpy as np
import sympy as sp
from dolfin import *
import itertools


def relu(x):
    '''Vectorized RELU'''
    return np.maximum(0, x)


def hat(x, x0, a=0, b=1):
    '''    /\              y = 1      
    ------/  \-------          0
         a x0 b
    '''
    a, b = float(a), float(b)
    assert a < x0 < b
    # To reference domain which is (0, 1)
    x = (x-a)/(b-a)
    x0 = (x0-a)/(b-a)
    return relu(x)/x0 - relu(x-x0)/x0/(1-x0) + relu(x-1)/(1-x0)


def hat_sympy(x, x0, a=0, b=1):
    '''Sympy version'''
    x, x0_, a_, b_ = sp.symbols('x x0 a b')
    x = (x - a_)/(b_-a_)
    x0_ = (x0 - a_)/(b_-a_)
    f = sp.Max(0, x)/x0_ - sp.Max(0, x-x0_)/x0_/(1-x0_) + sp.Max(0, x-1)/(1-x0_)

    return f.subs({a_: a, b_: b, x0_: x0})
            

def odd_tooth(x, x0=-0.5, x1=0.5):
    '''v^: [-1, 1] -> [-1, 1]
           /\            y = 1
    ---\  /  \-------        0
        \/                  -1
     -1 x0  x1
    '''
    assert -1 < x0 < x1 < 1
    return -1.0*hat(x, x0=x0, a=-1, b=0) + 1.0*hat(x, x0=x1, a=0, b=1)


def odd_tooth_sympy(x, x0=-0.5, x1=0.5):
    '''Sympy version'''
    return -1.0*hat_sympy(x, x0=x0, a=-1, b=0) + 1.0*hat_sympy(x, x0=x1, a=0, b=1)


def even_tooth(x, x0=-2/3., x1=2/3.):
    '''^v^: [-1, 1] -> [-1, 1]
        /\    /\         y = 1
    ---/  \  /  \---       = 0
           \/                -1
      -1 x0 0  x1 1
    '''
    assert -1 < x0 < x1 < 1
    
    return hat(x, x0=x0, a=-1, b=x0+(x0+1)) - hat(x, x0=0, a=x0+(x0+1), b=x1-(1-x1))+ hat(x, x0=x1, a=x1-(1-x1), b=1)


def even_tooth_sympy(x, x0=-2/3., x1=2/3.):
    '''Sympy version'''
    return hat_sympy(x, x0=x0, a=-1, b=x0+(x0+1)) - hat_sympy(x, x0=0, a=x0+(x0+1), b=x1-(1-x1))+ hat_sympy(x, x0=x1, a=x1-(1-x1), b=1)


def odd_sawtooth(x, s):
    ''' v^ Composition'''
    assert s >= 1
    if s == 1:
        return odd_tooth(x)
    return odd_sawtooth(odd_tooth(x), s-1)


def even_sawtooth(x, s):
    ''' ^v^ Composition'''
    assert s >= 1
    if s == 1:
        return even_tooth(x)
    return even_sawtooth(even_tooth(x), s-1)


def eval_hierarchy(x, depth, basis):
    '''Compositions of functions in basis @ x'''
    values = [[f(x) for f in basis]]
    for k in range(1, depth):
        values.append([f(y) for y in values[-1] for f in basis])
    return values


def get_composed(composition, basis, degree=5):
    '''Composition encodes the function made of basis. Return Expression for it.'''
    template = '''
class MyCppExpression%(compose)s : public Expression
{
public:
  void eval(Array<double>& values, const Array<double>& x) const
  {
    %(body)s;
    values[0] = %(final)s;
  }
};'''
    
    # Univariate with everything but 'x' specified
    replacements = []
    for f in basis:
        replace, = f.free_symbols
        replacements.append(str(replace))

    body = []
    prev = sp.Symbol('x[0]')
    for k, c in enumerate(reversed(composition)):
        body.append('const double y%d = %s;' % (k, sp.printing.ccode(basis[c].subs(replacements[c], prev))))
        prev = sp.Symbol('y%d' % k)

    body = '\n'.join(body)
    final = str(prev)
    compose = ''.join(map(str, composition))

    code = template % {'compose': compose, 'body': body, 'final': final}

    return Expression(code, degree=5)


def get_level(level, basis, degree=5):
    '''Compositions of level length of the basis functions'''
    numbs = list(range(len(basis)))
    for composition in itertools.product(*[numbs]*level):
        yield get_composed(composition, basis, degree=degree)


def get_all_levels(max_level, basis, degree=5):
    '''Up to'''
    for f in itertools.chain(*[get_level(l, basis, degree) for l in range(1, max_level+1)]):
        yield f


def get_perlevel(counts, basis, degree=5):
    '''From the complete hierarchy len(counts) use counts[i] functions per level'''
    all_basis = []
    for level, count in enumerate(counts, 1):
        foos = list(get_level(level, basis, degree))
        # Entire level
        if count == -1:
            all_basis.extend(foos)
        # Pick
        else:
            while count > 0:
                idx = np.random.randint(0, len(foos))
                all_basis.append(foos.pop(idx))
                count -= 1

    return all_basis 


def solve_poisson_galerkin(basis_expr, mesh, f):
    '''Solve -Delta u = f with Galerking method using basis of basis_expr'''
    # Get basis foos as FEM mesh foos
    S = FunctionSpace(mesh, 'CG', 10)
    # So that we can compute their gradient
    basis = [interpolate(b, S) for b in basis_expr]
    
    # Build approximation space
    V = VectorFunctionSpace(mesh, 'R', 0, len(basis))
    u, v = TrialFunction(V), TestFunction(V)
    
    u = sum(ui*fi for ui, fi in zip(u, basis))
    v = sum(vi*fi for vi, fi in zip(v, basis))

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    A, b = map(assemble, (a, L))

    A_ = A.array()
    print('System conditioning', np.linalg.cond(A_))
    
    ch = Function(V)
    solve(A, ch.vector(), b)

    wh = sum(chi*fi for whi, fi in zip(ch, basis))
    
    return ch.vector().get_local(), wh
        
# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Plot functions for each level
    if False:
        x = np.linspace(-1, 1, 10001)

        m = 3

        hierarchy = eval_hierarchy(depth=m, basis=(odd_tooth, even_tooth), x=x)
        basis_names = ('O', 'E')
        level_names = [['O', 'E']]
        for level in hierarchy:
            fig, axarr = plt.subplots(len(level)/2, 2, sharex=True, sharey=True)
            axarr = axarr if len(axarr.shape) == 1 else itertools.chain(*axarr)

            names = level_names.pop()

            for ax, f, name in zip(axarr, level, names):
                ax.plot(x, f)
                ax.set_title(name)

            level_names.append([' o '.join([f, prev]) for prev in names for f in basis_names])

            
    # NOTE: v^ with weights from yarotsky approximates x**2 wave
    if False:
        x = np.linspace(-1, 1, 10000)
        
        y0 = np.where(x < 0, x*(x+1), x*(1-x))

        # To verify the claim we check the (exponential) decay of the approximation
        # error
        ms = np.arange(2, 11)
        errors = []
        for m in ms:
            y = np.zeros_like(y0)
            x_ = x
            for k in range(1, 1+m):
                x_ = odd_tooth(x_)
                y += x_/(2**(2*k))
            
            e = np.linalg.norm(y0 - y, np.inf)
            errors.append(e)

        c0, c1 = np.polyfit(ms, np.log2(errors), deg=1)
    
        plt.figure()
        plt.semilogy(ms, errors, basey=2, linestyle='none', marker='o')
        plt.semilogy(ms, 2**(c0*ms + c1), basey=2)
        plt.xlabel('nlevels')
        plt.ylabel('error')


    # Check basis functions as expressions
    if False:
        x = sp.Symbol('x')
        basis = [odd_tooth_sympy(x), even_tooth_sympy(x)]
        f = get_composed(composition=(1, 0, 0), basis=basis)

        x = np.linspace(-1, 1, 10000)
        y = map(f, x)
        plt.figure()
        plt.plot(x, y)

    # Finally the fun part - using the basis
    if False:
        nlevels = 5

        x = sp.Symbol('x')
        basis = [odd_tooth_sympy(x), even_tooth_sympy(x)]
        all_basis = list(get_all_levels(nlevels, basis=basis))
        # Integration mesh
        mesh = IntervalMesh(200000, -1, 1)
        # Solve Poisson problem
        k = 1

        f = Expression('(k*pi)*(k*pi)*sin(k*pi*x[0])', degree=5, k=k)
        u = Expression('sin(k*pi*x[0])', degree=5, k=k)

        # Solution as expression
        ch, uh = solve_poisson_galerkin(all_basis, mesh, f)    
        # As function (for plotting and what not)
        Vh = FunctionSpace(mesh, 'CG', 3)
        uh = project(uh, Vh)
        error = errornorm(u, uh, 'H10')

        # Plot
        xh = Vh.tabulate_dof_coordinates().reshape((-1, ))
        uh = uh.vector().get_local()
        # Order dofs according to growing x
        idx = np.argsort(xh)
        xh, uh = xh[idx], uh[idx]

        print('Error with %d basis functions %g' % (len(all_basis), error))

        plt.figure()
        plt.plot(xh, uh)
        plt.plot(xh, map(u, xh))


    # Per level
    x = sp.Symbol('x')
    basis = [odd_tooth_sympy(x), even_tooth_sympy(x)]
    all_basis = get_perlevel((-1, -1, -1), basis=basis, degree=5)
    # Integration mesh
    mesh = IntervalMesh(200000, -1, 1)
    # Solve Poisson problem
    k = 1
    
    f = Expression('(k*pi)*(k*pi)*sin(k*pi*x[0])', degree=5, k=k)
    u = Expression('sin(k*pi*x[0])', degree=5, k=k)
    
    # Solution as expression
    uh = solve_poisson_galerkin(all_basis, mesh, f)    
    # As function (for plotting and what not)
    Vh = FunctionSpace(mesh, 'CG', 3)
    ch, uh = project(uh, Vh)
    error = errornorm(u, uh, 'H10')
    
    # Plot
    xh = Vh.tabulate_dof_coordinates().reshape((-1, ))
    uh = uh.vector().get_local()
    # Order dofs according to growing x
    idx = np.argsort(xh)
    xh, uh = xh[idx], uh[idx]

    print('Error with %d basis functions %g' % (len(all_basis), error))
    
    plt.figure()
    plt.plot(xh, uh)
    plt.plot(xh, map(u, xh))

    
    # Add AMG results


    plt.show()
