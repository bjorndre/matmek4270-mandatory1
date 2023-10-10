import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xij, self.yij ...
        self.N = N
        self.dx = self.L / N
        x = np.linspace(0, self.L, self.N+1)
        y = np.linspace(0, self.L, self.N+1)
        self.xij, self.yij = np.meshgrid(x,y, indexing="ij")
        return self.xij, self.yij

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= self.dx**2
        return D

    def laplace(self):
        """Return vectorized Laplace operator"""
        D2 = self.D2()
        return (sparse.kron(D2, sparse.eye(self.N+1)) + sparse.kron(sparse.eye(self.N+1), D2))

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N+1, self.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel()==1)[0]
        return bnds

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        # return A, b
        A = self.laplace()
        bnds = self.get_boundary_indices()

        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()

        F = sp.lambdify((x, y), self.f)(self.xij, self.yij)
        b = F.ravel()
        b_bnd = sp.lambdify((x, y), self.ue)(self.xij, self.yij)
        b[bnds] = b_bnd.ravel()[bnds]
        #b[0:100] = 2*b[0:100]/101
        #b[-101:] = 2*b[-101:]/101
        #b_f = sp.lambdify((x, y), bnd_func)(xij,yij)
        #B_F = b_f.ravel()
        
        #b[bnds] = self.ue()#B_F[bnds]
        return A, b
    
    def l2_error(self, u):
        """Return l2-error norm"""
        uj = sp.lambdify((x,y) , self.ue)(self.xij,self.yij)
        return np.sqrt(self.dx*self.dx*np.sum((u-uj)**2))

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.dx)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, xi, yj):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        clo_x = int(min(enumerate(self.xij[:,0]),  key=lambda s: abs(s[1]-xi))[0])
        clo_y = int(min(enumerate(self.yij[0,:]),  key=lambda s: abs(s[1]-yj))[0])
        
        if self.xij[clo_x, 0] != xi or self.yij[0, clo_y] != yj:
            #Checks if x or y lies close to the boundary
            if clo_x in [0,1]:
                lx = self.Lagrangebasis(self.xij[:clo_x+2, 0], x=x)
                x_dim = [0, clo_x+2]
            elif clo_x in [len(self.xij)-1, len(self.xij)-2]:
                lx = self.Lagrangebasis(self.xij[clo_x-1:, 0], x=x)
                x_dim = [clo_x-1, None]
            else:
                lx = self.Lagrangebasis(self.xij[clo_x-1:clo_x+2, 0], x=x)
                x_dim = [clo_x-1, clo_x+2]

            if clo_y in [0,1]:
                ly = self.Lagrangebasis(self.yij[0, :clo_y+2], x=y)
                y_dim = [0, clo_y+2]
            elif clo_y in [len(self.yij)-1, len(self.yij)-2]:
                ly = self.Lagrangebasis(self.yij[0, clo_y-1:], x=y)
                y_dim = [clo_y-1, None]
            else:
                ly = self.Lagrangebasis(self.yij[0, clo_y-1:clo_y+2], x=y)
                y_dim = [clo_y-1, clo_y+2]
            
            if x_dim[1] == None and y_dim[1] == None:
                L2 = self.Lagrangefunction(self.U[x_dim[0]:, y_dim[0]:], lx, ly)
            elif x_dim[1] == None:
                L2 = self.Lagrangefunction(self.U[x_dim[0]:, y_dim[0]:y_dim[1]], lx, ly)
            elif y_dim[1] == None:
                L2 = self.Lagrangefunction(self.U[x_dim[0]:x_dim[1], y_dim[0]:], lx, ly)
            else:
                L2 = self.Lagrangefunction(self.U[x_dim[0]:x_dim[1], y_dim[0]:y_dim[1]], lx, ly)
            
            ans = L2.subs({x: xi, y: yj})
        
        elif self.xij[clo_x, 0] == xi and self.yij[0, clo_y] == yj:
            ans = self.U[clo_x, clo_y]
        
        return ans


    def Lagrangebasis(self, xj, x=x):
        n = len(xj)
        ell = []
        numert = sp.Mul(*[x - xj[i] for i in range(n)])

        for i in range(n):
            numer = numert/(x-xj[i])
            denom = sp.Mul(*[(xj[i] - xj[j]) for j in range(n) if i != j])
            ell.append(numer/denom)
        return ell
    
    def Lagrangefunction(self, u, basisx, basisy):
        N, M = u.shape
        f = 0
        for i in range(N):
            for j in range(M):
                f += basisx[i]*basisy[j]*u[i,j]
        return f
    
def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.dx/2, 1-sol.dx/2) - ue.subs({x: sol.dx, y: 1-sol.dx/2}).n()) < 1e-3

test_convergence_poisson2d()
test_interpolation()