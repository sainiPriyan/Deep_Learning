import numpy as np

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
y  = np.array([0, 0, 0, 1])

w1, w2, b = 0.1, 0.1, 0.1
lr = 2
epochs = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

for epoch in range(epochs+1):
    cost = 0

    
    if epoch % 100 == 0:
            print(f"Epoch {epoch}")

    for i in range(len(x1)):

        z = w1*x1[i] + w2*x2[i] + b
        y_hat = sigmoid(z)

        error = y_hat - y[i]
        cost += error**2

        dz = error * d_sigmoid(z)
        dw1 = dz * x1[i]
        dw2 = dz * x2[i]
        db  = dz

        w1 -= lr * dw1
        w2 -= lr * dw2
        b  -= lr * db

        if epoch % 100 == 0:
            print(f"x1={x1[i]}, x2={x2[i]}, y={y[i]}, y_hat={y_hat}, error={error}")
            print(f"w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")

    cost /= len(x1)

    if epoch % 100 == 0:
            print(f"Cost: {cost:.4f}")
    