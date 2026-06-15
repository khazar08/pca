# scalar helper
def sqrt(x):
    if x < 0:
        raise ValueError("sqrt of a negative number")
    if x == 0:
        return 0.0
    g = x if x >= 1.0 else 1.0          # crude but safe starting guess
    for _ in range(100):
        new_g = 0.5 * (g + x / g)        # Newton step on f(g) = g^2 - x
        if abs(new_g - g) < 1e-15:
            break
        g = new_g
    return g


# matrix helpers

def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]

def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)]
            for i in range(n)]


def column_means(X):
    n, d = len(X), len(X[0])
    return [sum(X[i][j] for i in range(n)) / n for j in range(d)]


def mean_center(X):
    means = column_means(X)
    Xc = [[X[i][j] - means[j] for j in range(len(X[0]))] for i in range(len(X))]
    return Xc, means


def covariance_matrix(Xc):
    """Sample covariance (divides by n-1) of already-centered data."""
    n, d = len(Xc), len(Xc[0])
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            s = sum(Xc[k][i] * Xc[k][j] for k in range(n)) / (n - 1)
            cov[i][j] = s
            cov[j][i] = s
    return cov


def jacobi_eigen(A, max_sweeps=100, tol=1e-12):
    n = len(A)
    A = [row[:] for row in A]                       # work on a copy
    V = [[1.0 if i == j else 0.0 for j in range(n)] # accumulates rotations
         for i in range(n)]

    for _ in range(max_sweeps):
        # locate the largest off-diagonal element
        p, q, off = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > off:
                    off, p, q = abs(A[i][j]), i, j
        if off < tol:
            break

        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        tau = (aqq - app) / (2.0 * apq)
        if tau >= 0:
            t = 1.0 / (tau + sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + sqrt(1.0 + tau * tau))
        c = 1.0 / sqrt(1.0 + t * t)
        s = t * c

        # update the two pivot diagonal entries, zero the off-diagonal
        A[p][p] = app - t * apq
        A[q][q] = aqq + t * apq
        A[p][q] = A[q][p] = 0.0

        # rotate the remaining entries of rows/cols p and q
        for i in range(n):
            if i != p and i != q:
                aip, aiq = A[i][p], A[i][q]
                A[i][p] = A[p][i] = c * aip - s * aiq
                A[i][q] = A[q][i] = s * aip + c * aiq

        # accumulate the rotation into the eigenvector matrix
        for i in range(n):
            vip, viq = V[i][p], V[i][q]
            V[i][p] = c * vip - s * viq
            V[i][q] = s * vip + c * viq

    eigenvalues = [A[i][i] for i in range(n)]
    eigenvectors = [[V[i][k] for i in range(n)] for k in range(n)]  # rows = vecs
    return eigenvalues, eigenvectors


def pca(X, n_components=None):
    Xc, means = mean_center(X)
    cov = covariance_matrix(Xc)
    eigvals, eigvecs = jacobi_eigen(cov)

    # sort components by descending eigenvalue
    order = sorted(range(len(eigvals)), key=lambda k: eigvals[k], reverse=True)
    eigvals = [eigvals[k] for k in order]
    eigvecs = [eigvecs[k] for k in order]

    # deterministic sign: make each vector's largest-magnitude entry positive
    for vec in eigvecs:
        lead = max(range(len(vec)), key=lambda i: abs(vec[i]))
        if vec[lead] < 0:
            for i in range(len(vec)):
                vec[i] = -vec[i]

    total = sum(max(v, 0.0) for v in eigvals) or 1.0  # over ALL components

    if n_components is None:
        n_components = len(eigvals)
    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:n_components]
    projected = [[sum(Xc[r][f] * comp[f] for f in range(len(comp)))
                  for comp in eigvecs] for r in range(len(Xc))]

    return {
        "projected": projected,
        "components": eigvecs,
        "explained_variance": eigvals,
        "explained_ratio": [v / total for v in eigvals],
        "mean": means,
    }
