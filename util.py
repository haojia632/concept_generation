import math

def sigmoid(x):
    return [1 / (1 + math.exp(-i)) for i in x]


if __name__ == "__main__":
    a = range(10)
    
    print(sigmoid(a))