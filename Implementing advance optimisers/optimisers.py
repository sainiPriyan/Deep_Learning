import numpy as np

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([0,0,0,1]) 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

def train(optimizer="sgd", epochs=101, lr=0.1):

    print(f'{optimizer.upper()}:')    

    w = np.array([0.1, 0.1])
    b = 0.1

    vw = np.zeros_like(w)
    vb = 0.0
    mw = np.zeros_like(w)
    mb = 0.0

    beta = 0.9        
    beta1 = 0.9       
    beta2 = 0.999     
    eps = 1e-8
    t = 0

    for epoch in range(epochs):
        idx = np.random.permutation(len(X))
        cost = 0.0   

        for i in idx:  
            z = np.dot(w, X[i]) + b
            y_hat = sigmoid(z)

            error = y_hat - y[i]
            cost += error**2
            dz = error * d_sigmoid(z)

            dw = dz * X[i]
            db = dz

            
            if optimizer == "sgd":
                w -= lr * dw
                b -= lr * db

           
            elif optimizer == "rmsprop":
                vw = beta * vw + (1 - beta) * dw**2
                vb = beta * vb + (1 - beta) * db**2
                w -= lr * dw / (np.sqrt(vw) + eps)
                b -= lr * db / (np.sqrt(vb) + eps)

            
            elif optimizer == "adam":
                t += 1
                mw = beta1 * mw + (1 - beta1) * dw
                mb = beta1 * mb + (1 - beta1) * db
                vw = beta2 * vw + (1 - beta2) * dw**2
                vb = beta2 * vb + (1 - beta2) * db**2

                mw_hat = mw / (1 - beta1**t)
                mb_hat = mb / (1 - beta1**t)
                vw_hat = vw / (1 - beta2**t)
                vb_hat = vb / (1 - beta2**t)

                w -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
                b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        cost /= len(X)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}  w={w}, b={b} Cost: {cost:.4f}")

    return w, b


train("sgd")
train("rmsprop")
train("adam")