import numpy as np

def cubic_spline(X,Y,X_target,ddy=(0,0)):
    n1 = len(X)-1
    n2 = len(X_target)
    h      = np.diff(X)
    delta  = np.diff(Y)
    Lambda = np.zeros(n1)
    Mu     = np.zeros(n1)
    g      = np.zeros(n1+1)
    M      = np.zeros(n1+1)
    result = np.zeros(n2)

    for i in range(1,n1):
        Mu[i]     = h[i-1]/(h[i]+h[i-1])
        Lambda[i] = h[i]  /(h[i]+h[i-1])
        g[i]      = 6*(delta[i]/h[i] - delta[i-1]/h[i-1])/(h[i-1]+h[i])
    
        M[0],M[-1]=ddy
        g[1]  -= Mu[1]     *M[0]
        g[-1] -= Lambda[-1]*M[-1]
        matrix = 2*np.identity(n1-1)
        matrix+= np.diag(Mu[2:]      ,k=-1)
        matrix+= np.diag(Lambda[1:-1],k= 1)
        solution = np.linalg.inv(matrix)
        solution = np.dot(solution,g[1:-1])
        M[1:n1]= solution

    for k in range(n2):
        index = np.abs(X-X_target[k]).argmin()
        if X[index]>=X_target[k] and index!=0:
            index-=1
        x = X_target[k]
        x0,y0 = X[index],Y[index]

        a0 = y0
        a1 = delta[index]/h[index] - (2*M[index]+M[index+1])*h[index]/6
        a2 = M[index]/2
        a3 = (M[index+1]-M[index])/h[index]/6

        result[k] = a0 + a1*(x-x0) + a2*(x-x0)**2 + a3*(x-x0)**3
    
    return result