
import numpy as np

eigenvalues = np.array([3, 1, 4, 2])
print('Eigenvalues &:')
print(np.argsort(eigenvalues)) 

print('Eigenvalues &&:')
print(np.argsort(eigenvalues)[::-1]) 

print('-' * 40)
print(eigenvalues[np.argsort(eigenvalues)[::-1]])