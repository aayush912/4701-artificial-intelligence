import numpy as np


def p1(k: int) -> str:
    # CREATE AN ARRAY AND KEEP STORING THE FACTORIALS UNTIL THE GIVEN INPUT.
    arr = [0] * k
    arr [0] = 1
    for i in range (1, k):
        arr [i] = arr [i-1] * (i+1)
    
    # STORE THE RESULT (IN REVERSE ORDER) IN A STRING USING A COMMA AS SEPARATOR.  REMOVE THE LAST COMMA AND RETURN THE STRING.
    s = ""
    for i in range (k-1, -1, -1):
        s = s + str (arr [i]) + ','
    return s[:-1]



def p2_a(x: list, y: list) -> list:
    # CREATE A COPY OF THE INPUT LIST SO THAT THE ORIGINAL IS NOT CHANGED. SORT IT IN DESCENDING ORDER, REMOVE THE LAST ELEMENT AND RETURN THE LIST.
    Y = y.copy()
    Y.sort(reverse=True)
    return Y[:-1]


def p2_b(x: list, y: list) -> list:
    # CREATE A COPY OF THE INPUT LIST SO THAT THE ORIGINAL IS NOT CHANGED.  REVERSE THE LIST AND RETURN IT.
    X = x.copy()
    X.reverse()
    return X


def p2_c(x: list, y: list) -> list:
    # APPEND THE TWO INPUT LISTS INTO ANOTHER EMPTY LIST AND SORT IT.
    Z = x + y
    Z.sort()

    # CREATE AN EMPTY LIST.  ITERATE THE APPENDED LIST PREVIOUSLY AND IF A NUMBER IS NOT PRESENT IN THE NEW LIST:
    # APPEND IT TO THE NEW LIST ELSE SKIP IT.
    # RETURN THE NEW LIST.
    res = []
    for i in range (len (Z)):
        if Z[i] not in res:
            res.append (Z[i])
    return res


def p2_d(x: list, y: list) -> list:
    # ADD THE TWO INPUT LISTS TO AN EMPTY LIST AS ITS ELEMENTS 
    # AND RETURN THE NEW LIST. 
    P = [x,y]
    return P


def p3_a(x: set, y: set, z: set) -> set:
    # RETURN UNION OF THE THREE SETS
    return x.union(y).union(z)


def p3_b(x: set, y: set, z: set) -> set:
    # RETURN INTERSECTION OF THE THREE SETS
    return x.intersection(y).intersection(z)


def p3_c(x: set, y: set, z: set) -> set:
    # CREATE AN EMPTY LIST.  
    # ITERATE EACH SET ONCE AND IF A NUMBER IN THE SET IS NOT PRESENT IN THE NEW LIST, ADD IT TO THE LIST. 
    # RETURN THE LIST AS A SET.
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
    # USE NUMPY TO CREATE A 2D ARRAY OF THE GIVEN SET-UP OF THE 5X5 BOARD.
    # RETURN THE 2D ARRAY.
    arr = np.array([[1,1,1,1,1],[1,0,0,0,1],[1,0,2,0,1],[1,0,0,0,1],[1,1,1,1,1]])
    return arr


def p4_b(x: np.array) -> list:
    # FIND THE POSITION OF THE PAWN
    p, q, n = -1, -1, len(x)
    for i in range (n):
        for j in range (n):
            if x[i][j] == 2:
                p = i
                q = j
                break
    
    # CHECK IN ALL POSSIBLE 8 DIRECTIONS FOR A KNIGHT THAT CAN ATTACK THE PAWN.
    X = [-2, -2, -1, 1, -1, 1, 2, 2]
    Y = [-1, 1, -2, -2, 2, 2, -1, 1]
    res = []
    for i in range (8):
        newX = p + X[i]
        newY = q + Y[i]
        # CHECK IF THE NEW CELL IS VALID.
        # ONLY VALID CELLS ARE APPENDED TO THE RESULT.
        if (isValid (newX, newY, x)):
            res.append ((newX, newY))
    return res


def isValid(x, y, matrix):
    # VALID CELL CRITERIA:
    # CELL LIES WITHIN THE DIMENSIONS OF THE BOARD
    # AND CELL CONTAINS A KNIGHT
    if x >= 0 and x < len(matrix) and y >= 0 and y < len(matrix) and matrix [x][y] == 1:
        return True
    else:
        return False


def p5_a(x: dict) -> int:
    # ITERATE THE DICTIONARY AND CHECK IF A KEY (NODE) HAS VALUES (CHILDREN).
    # IF FALSE, THEN IT IS AN ISOLATED NODE, SO INCREMENT COUNT.
    # RETURN COUNT.
    res = 0
    for node in x:
        if x[node] is None or len(x[node]) == 0:
            res = res + 1
    return res


def p5_b(x: dict) -> int:
    # NUMBER OF NON-ISOLATED NODES = TOTAL NODES - ISOLATED NODES.
    # TOTAL NODES = SIZE OF DICTIONARY
    # ISOLATED NODES = CALCULATED IN FUNCTION (p5_a)
    # HENCE, RETURN [LENGTH OF DICTIONARY - RESULT RETURNED BY FUNCTION (p5_a)]
    return len (x) - p5_a(x)


def p5_c(x: dict) -> list:
    # CREATE A SET OF VISITED NODES.  
    # CREATE AN EMPTY LIST TO STORE THE RESULT.  
    seen = set()
    res = []
    for node in x:
        
        # WHENEVER YOU EXPLORE AN UNVISITED NODE, ADD IT TO THE VISITED SET.
        seen.add (node)

        # WHEN EXPLORING THE CHILDREN NODES, IF A NODE IS NOT VISITED, APPEND THE TUPLE OF THE PARENT AND CHILD NODE TO THE RESULT LIST.
        for child in x[node]:
            if child not in seen:
                res.append ((node, child))

    # RETURN THE LIST.
    return res


def p5_d(x: dict) -> np.array:
    # SIZE OF ADJACENCY MATRIX = N X N, WHERE N IS THE NUMBER OF NODES (= SIZE OF DICTIONARY).
    # CREATE A MATRIX USING NUMPY.
    n = len(x)
    adjMatrix = np.empty([n,n], int)

    # SET ALL THE CELLS TO 0 INITIALLY
    for i in range (n):
        for j in range (n):
            adjMatrix [i][j] = 0
    
    # FOR EVERY KEY (NODE) IN THE DICTIONARY:
    # ITERATE ITS ADJACENCY LIST AND UPDATE THE VALUE OF THE MATRIX TO 1 FOR EVERY NODE WHICH IS CONNECTED TO IT.
    for node in x:
        for i in range (len (x[node])):
            adjMatrix [ord(node)-65][ord(x[node][i])-65] = 1
    
    # RETURN THE ADJACENCY MATRIX
    return adjMatrix
    


class PriorityQueue(object):
    def __init__(self):
        # CREATE AN EMPTY LIST TO BE USED AS A PRIORITY QUEUE.
        self.heap = []

    def push(self, x):
        # CREATE A DICTIONARY OF THE PRODUCTS AS KEYS AND PRICES AS VALUES.
        rates = {'apple': 5.0, 
                 'banana': 4.5, 
                 'carrot': 3.3,
                 'kiwi': 7.4,
                 'orange': 5.0,
                 'mango': 9.1,
                 'pineapple': 9.1}
        
        # APPEND THE PRODUCT AND ITS PRICE AS A TUPLE TO THE PRIORITY QUEUE.
        self.heap.append((x, rates[x]))

    def pop(self):
        # SORT THE PRODUCTS ON THE BASIS OF ITS PRICES IN DESCENDING ORDER.
        # POP OUT THE FIRST (TOP) PRODUCT AND RETURN IT.
        self.heap.sort(key=lambda tup: tup[1], reverse=True)
        return self.heap.pop(0)[0]

    def is_empty(self):
        # RETURN TRUE IF THE SIZE OF THE PRIORITY QUEUE IS 0.
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