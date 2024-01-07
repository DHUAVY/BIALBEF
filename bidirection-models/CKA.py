import numpy as np

class CKALoss:

    def __init__(self, X, Y, lambda_CKA = 0.01, kernel=None):
        self.X = X
        self.Y = Y
        self.lambda_CKA = 0.01
        self.kernel = kernel
        
    def linear_kernel(self, X, Y):
        return np.matmul(X, Y.T)

    def rbf(self, sigma=None):
        """
        Radial-Basis Function kernel for X and Y with bandwith chosen
        from median if not specified.
        """
        GX = np.dot(X, Y.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def HSIC(self, K, L):
        """
        Calculate Hilbert-Schmidt Independence Criterion on K and L.
        """
        n = K.shape[0]
        H = np.identity(n) - (1./n) * np.ones((n, n))

        KH = np.matmul(K, H)
        LH = np.matmul(L, H)
        return 1./((n-1)**2) * np.trace(np.matmul(KH, LH))

    def CKA(self):
        """
        Calculate Centered Kernel Alingment for X and Y. If no kernel
        is specified, the linear kernel will be used.
        """
        kernel = self.linear_kernel if self.kernel is None else self.kernel
        
        K = kernel(self.X, self.X)
        L = kernel(self.Y, self.Y)
            
        hsic = self.HSIC(K, L)
        varK = np.sqrt(self.HSIC(K, K))
        varL = np.sqrt(self.HSIC(L, L))
        return hsic / (varK * varL)