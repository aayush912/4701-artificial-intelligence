import numpy as np
import sys

def main():
    input_file = np.genfromtxt(sys.argv[1], delimiter = ',')
    output_file = str(sys.argv[2])
    data = input_file[:,:2]
    y = input_file[:,2]
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

    normalized_data = (data - data.mean(axis=0))/ data.std(axis=0)

    normalized_data = np.hstack([np.ones((normalized_data.shape[0], 1)), normalized_data])

    iterations = 100 
    for alpha in alphas:
        weights = [0, 0, 0]
        for i in range(iterations):
            weight0 = calculate(0, alpha, weights, normalized_data, y)
            weight1 = calculate(1, alpha, weights, normalized_data, y)
            weight2 = calculate(2, alpha, weights, normalized_data, y)
            weights = [weight0, weight1, weight2]
        with open(output_file, "a") as f:
            f.write(str(alpha) + ", ")
            f.write(str(iterations) + ", ")
            f.write(str(weights[0]) + ", ")
            f.write(str(weights[1]) + ", ") 
            f.write(str(weights[2]) + ", ")
            f.write("\n")

    my_alpha = 0.75
    my_iterations = 30
    for i in range(my_iterations):
        weight0 = calculate(0, my_alpha, weights, normalized_data, y)
        weight1 = calculate(1, my_alpha, weights, normalized_data, y)
        weight2 = calculate(2, my_alpha, weights, normalized_data, y)
        weights = [weight0, weight1, weight2]
    with open(output_file, "a") as f:
        f.write(str(my_alpha) + ", ")
        f.write(str(iterations) + ", ")
        f.write(str(weights[0]) + ", ")
        f.write(str(weights[1]) + ", ") 
        f.write(str(weights[2]) + ", ")
        f.write("\n")

def calculate(beta, alpha, weights, data, y):
    total = 0
    for i, row in enumerate(data):
        fx = np.dot(row, weights)
        error = fx - y[i]
        total = total + (error * row[beta])
    val = (alpha * total)/len(data)
    return beta - val
    

if __name__ == '__main__':
    main()