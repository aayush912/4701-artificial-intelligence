import numpy as np


def p1(k: int) -> str:
    arr = [0] * k
    arr [0] = 1
    for i in range (1, k):
        arr [i] = arr [i-1] * (i+1)
    s = ""
    for i in range (k-1, -1, -1):
        s = s + str (arr [i]) + ','
    return s[:-1]



def p2_a(x: list, y: list) -> list:
    Y = y.copy()
    Y.sort(reverse=True)
    return Y[:-1]


def p2_b(x: list, y: list) -> list:
    X = x.copy()
    X.reverse()
    return X


def p2_c(x: list, y: list) -> list:
    Z = x + y
    Z.sort()
    res = []
    for i in range (len (Z)):
        if Z[i] not in res:
            res.append (Z[i])
    return res


def p2_d(x: list, y: list) -> list:
    P = [x,y]
    return P


def p3_a(x: set, y: set, z: set) -> set:
    return x.union(y).union(z)


def p3_b(x: set, y: set, z: set) -> set:
    return x.intersection(y).intersection(z)


def p3_c(x: set, y: set, z: set) -> set:
    res = []
    for var in x:
        if var not in y and var not in z:
            res.append(var)

    for var in y:
        if var not in x and var not in z:
            res.append(var)
    
    for var in z:
        if var not in x and var not in y:
            res.append(var)
    
    return (set (res))


def p4_a() -> np.array:
    arr = np.array([[1,1,1,1,1],[1,0,0,0,1],[1,0,2,0,1],[1,0,0,0,1],[1,1,1,1,1]])
    return arr


def p4_b(x: np.array) -> list:
    p, q = -1, -1
    for i in range (5):
        for j in range (5):
            if x[i][j] == 2:
                p = i
                q = j
                break
    
    X = [-2, -2, -1, 1, -1, 1, 2, 2]
    Y = [-1, 1, -2, -2, 2, 2, -1, 1]
    res = []
    for i in range (8):
        newX = p + X[i]
        newY = q + Y[i]
        if (isValid (newX, newY, x)):
            res.append ((newX, newY))
    return res


def isValid(x, y, matrix):
    if x >= 0 and x < 5 and y >= 0 and y < 5 and matrix [x][y] == 1:
        return True
    else:
        return False


def p5_a(x: dict) -> int:
    res = 0
    for node in x:
        if x[node] is None or len(x[node]) == 0:
            res = res + 1
    return res


def p5_b(x: dict) -> int:
    return len (x) - p5_a(x)


def p5_c(x: dict) -> list:
    seen = set()
    res = []
    for node in x:
        seen.add (node)
        for child in x[node]:
            if child not in seen:
                res.append ((node, child))
    return res


def p5_d(x: dict) -> np.array:
    n = len(x)
    adjMatrix = np.empty([n,n], int)
    for i in range (n):
        for j in range (n):
            adjMatrix [i][j] = 0
    for node in x:
        for i in range (len (x[node])):
            adjMatrix [ord(node)-65][ord(x[node][i])-65] = 1
    return adjMatrix
    


class PriorityQueue(object):
    def __init__(self):
        self.heap = []

    def push(self, x):
        rates = {'apple': 5.0, 'banana': 4.5, 'carrot': 3.3, 'kiwi': 7.4, 'orange': 5.0, 'mango': 9.1, 'pineapple': 9.1}
        self.heap.append((x, rates[x]))

    def pop(self):
        self.heap.sort(key=lambda x: x[1], reverse=True)
        return self.heap.pop(0)[0]

    def is_empty(self):
        return len(self.heap) == 0


if __name__ == '__main__':
    print(p1(k=8))
    print('-----------------------------')
    print(p2_a(x=[], y=[1, 3, 5]))
    print(p2_b(x=[2, 4, 6], y=[]))
    print(p2_c(x=[1, 3, 5, 7], y=[1, 2, 5, 6]))
    print(p2_d(x=[1, 3, 5, 7], y=[1, 2, 5, 6]))
    print('------------------------------')
    print(p3_a(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print(p3_b(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print(p3_c(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print('------------------------------')
    print(p4_a())
    print(p4_b(p4_a()))
    print('------------------------------')
    graph = {
        'A': ['D', 'E'],
        'B': ['E', 'F'],
        'C': ['E'],
        'D': ['A', 'E'],
        'E': ['A', 'B', 'C', 'D'],
        'F': ['B'],
        'G': []
    }
    print(p5_a(graph))
    print(p5_b(graph))
    print(p5_c(graph))
    print(p5_d(graph))
    print('------------------------------')
    pq = PriorityQueue()
    pq.push('apple')
    pq.push('kiwi')
    pq.push('orange')
    while not pq.is_empty():
        print(pq.pop())