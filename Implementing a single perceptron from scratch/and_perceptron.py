import numpy as np

x1 = np.array([0,0,1,1])
x2 = np.array([0,1,0,1])
y = np.array([0,0,0,1])

w1, w2, b = 0.1,0.1,0.1
lr = 0.5
epochs = 10

def activation(z):
    return 1 if z > 0 else 0


for e in range(epochs):

    print(f'Epoch {e}:')

    cost = 0

    for i in range(len(x1)):

        z = w1*x1[i] + w2*x2[i] + b
        y_hat = activation(z)

        error = y[i] - y_hat
        cost += error**2

        w1 += lr*error*x1[i]
        w2 += lr*error*x2[i]
        b += lr*error

        print(f"x1={x1[i]}, x2={x2[i]}, y={y[i]}, y_hat={y_hat}, error={error}")
        print(f"w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")

    cost /= len(x1)
    print(f"Cost: {cost}")