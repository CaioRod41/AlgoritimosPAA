#Busca linear

def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

#Ordenação por seleção

def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

#Busca binária
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

#Algoritmo de Dijkstra

import heapq

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances


#Algoritmo de Strassen

import numpy as np

# Função para adicionar duas matrizes
def add_matrices(A, B):
    return np.add(A, B)

# Função para subtrair duas matrizes
def subtract_matrices(A, B):
    return np.subtract(A, B)

# Função para a multiplicação de matrizes pelo método de Strassen
def strassen_multiplication(A, B):
    # Caso base: matriz 1x1
    if len(A) == 1:
        return A * B
    
    # Dividindo a matriz A e B em submatrizes
    mid = len(A) // 2
    
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    # Calculando os 7 produtos de Strassen
    P1 = strassen_multiplication(A11, subtract_matrices(B12, B22))
    P2 = strassen_multiplication(add_matrices(A11, A12), B22)
    P3 = strassen_multiplication(add_matrices(A21, A22), B11)
    P4 = strassen_multiplication(A22, subtract_matrices(B21, B11))
    P5 = strassen_multiplication(add_matrices(A11, A22), add_matrices(B11, B22))
    P6 = strassen_multiplication(subtract_matrices(A12, A22), add_matrices(B21, B22))
    P7 = strassen_multiplication(subtract_matrices(A11, A21), add_matrices(B11, B12))
    
    # Combinando as submatrizes para formar a matriz C
    C11 = add_matrices(subtract_matrices(add_matrices(P5, P4), P2), P6)
    C12 = add_matrices(P1, P2)
    C21 = add_matrices(P3, P4)
    C22 = subtract_matrices(subtract_matrices(add_matrices(P5, P1), P3), P7)
    
    # Unindo as submatrizes C11, C12, C21, C22 em uma única matriz
    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    C = np.vstack((top, bottom))

    return C

# Função auxiliar para ajustar o tamanho das matrizes para serem quadradas e de dimensão 2^n
def pad_matrix(matrix, new_size):
    padded_matrix = np.zeros((new_size, new_size))
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix

# Exemplo de uso:
if __name__ == "__main__":
    A = np.random.randint(10, size=(4, 4))
    B = np.random.randint(10, size=(4, 4))

    # Verifica se as matrizes são quadradas e têm tamanho de 2^n
    assert A.shape == B.shape
    n = A.shape[0]
    
    # Chamada ao algoritmo de Strassen
    C = strassen_multiplication(A, B)

    print("Matriz A:")
    print(A)
    print("\nMatriz B:")
    print(B)
    print("\nMatriz C (resultado da multiplicação de Strassen):")
    print(C)
