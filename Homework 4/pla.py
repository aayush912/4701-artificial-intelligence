import sys
import numpy as np

def main ():
    input_file = np.genfromtxt(sys.argv[1], delimiter = ',')
    output_file = str(sys.argv[2])
    data = input_file[:, [0, 1]]
    bias = input_file[:, [2]]
    weights = perceptron(data, bias, output_file)

def perceptron (data, bias, output_file):
    rows = data.shape[1]
    weights = np.zeros((rows + 1, 1), dtype=float)
    convergence = False
    while convergence == False:
        convergence = True
        for i in range (data.shape[0]):
            updatedWeight = weights[-1] + np.dot(data[[i], :], weights[:-1, [0]])
            sgn = 1 if updatedWeight > 0 else -1
            if bias[i] * sgn <= 0:
                weights[-1] = weights [-1] + bias [i]
                weights[:-1, [0]] = weights [:-1, [0]] + bias [i] * np.transpose(data[[i], :])
                convergence = False
        if convergence == False:
            with open(output_file, "a") as f:
                for weight in weights:
                    f.write (str(weight[0]) +  ",")
                f.write("\n")
    return weights

if __name__ == '__main__':
    main()
