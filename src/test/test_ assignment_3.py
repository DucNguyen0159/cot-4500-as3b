import numpy as np

def gaussian_elimination():
    A = np.array([
        [2, -1, 1, 6],
        [1, 3, 1, 0],
        [-1, 5, 4, -3]
    ], dtype=float)
    n = len(A)
    
    for i in range(n):
        if A[i, i] == 0:
            for k in range(i + 1, n):
                if A[k, i] != 0:
                    A[[i, k]] = A[[k, i]]
                    break
        
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (A[i, -1] - np.dot(A[i, i+1:n], x[i+1:n])) / A[i, i]
    
    print("Solution:", x, "\n")

def lu_factorization():
    A = np.array([
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ], dtype=float)
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
    
    det = np.prod(np.diag(U))
    print("Determinant:", det, "\n")
    print("L:\n", L, "\n")
    print("U:\n", U, "\n")

def diagonally_dominate():
    A = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])
    for i in range(len(A)):
        if abs(A[i, i]) < sum(abs(A[i, :])) - abs(A[i, i]):
            return False
    return True

def is_symmetric(A):
    return np.allclose(A, A.T)

def sub_matrix(A, k):
    return A[:k, :k]

def determinant(A):
    return np.linalg.det(A)

def positive_definite():
    A = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])
    if not is_symmetric(A):
        return False
    for k in range(1, len(A) + 1):
        if determinant(sub_matrix(A, k)) <= 0:
            return False
    return True

if __name__ == "__main__":
    gaussian_elimination()
    lu_factorization()
    print("Diagonally dominant:", diagonally_dominate(), "\n")
    print("Positive definite:", positive_definite())
