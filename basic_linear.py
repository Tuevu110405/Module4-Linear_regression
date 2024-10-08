#dataset
import numpy as np
import matplotlib.pyplot as plt
import random

def get_column(data, index):
    return [row[index] for row in data]

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()
    N = len(data)

    #get tv
    tv_data = get_column(data, 0)

    #get radio
    radio_data = get_column(data, 1)

    #get newspaper
    newspaper_data = get_column(data, 2)

    #get sales
    sales_data = get_column(data, 3)

    #building X input and y output for training
    X = [tv_data, radio_data, newspaper_data]
    y = sales_data
    return X, y

#Q1
X, y = prepare_data('advertising.csv')
list = [sum(X[0][:5]), sum(X[1][:5]), sum(X[2][:5]), sum(y[:5])]
print(list)
print(type(X))


print(type(y))

def initialize_params():
    w1 , w2 , w3 , b = (0.016992259082509283 , 0.0070783670518262355 , -0.002307860847821344 , 0)
    return w1, w2, w3, b

def implement_linear_regression(X_data, y_data, epoch_max = 50, lr = 1e-5):
    losses = []

    w1, w2, w3, b = initialize_params()

    N = len(y_data)
    for epoch in range(epoch_max):
        for i in range(N):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]

            y = y_data[i]

            #compute output
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            #compute loss
            loss = compute_loss_mse(y, y_hat)

            #compute gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            #update parameters
            w1 = update_weight_wi( w1 , dl_dw1 , lr )
            w2 = update_weight_wi( w2 , dl_dw2 , lr )
            w3 = update_weight_wi( w3 , dl_dw3 , lr )
            b = update_weight_b( b , dl_db , lr )

            #logging
            losses.append(loss)

    return w1, w2, w3, b, losses

def predict(x1, x2, x3, w1, w2, w3, b):
    y_hat = w1*x1 + w2*x2 + w3*x3 + b
    return y_hat

y = predict ( x1 =1 , x2 =1 , x3 =1 , w1 =0 , w2 =0.5 , w3 =0 , b =0.5)
print(y)

def compute_loss_mse(y, y_hat):
    loss = np.float64((y - y_hat)**2)
    return loss

l = compute_loss_mse(y= 0.5, y_hat = 1)
print(l)

def compute_gradient_wi(x, y, y_hat):
    dl_dw = -2 * x * (y - y_hat)
    return dl_dw

g_wi = compute_gradient_wi( x =1.0 , y =1.0 , y_hat =0.5)
print ( g_wi )

def compute_gradient_b(y, y_hat):
    dl_db = -2 * (y - y_hat)
    return dl_db

g_b = compute_gradient_b( y =2.0 , y_hat =0.5)
print ( g_b )

#update weight
def update_weight_wi(w, dl_dw, lr):
    w = w - lr * dl_dw
    return w

def update_weight_b(b, dl_db, lr):
    b = b - lr * dl_db
    return b

after_wi = update_weight_wi(w =1.0 , dl_dw =-0.5 , lr =1e-5)
print(after_wi)

after_b = update_weight_b(b =0.5 , dl_db =-1.0 , lr =1e-5)
print(after_b)

(w1, w2, w3, b, losses) = implement_linear_regression(X, y)
plt.plot(losses[:100])
plt.xlabel('#iteration')
plt.ylabel('loss')
plt.show()

X,y = prepare_data('advertising.csv')
(w1, w2, w3, b, losses) = implement_linear_regression(X, y)
print(w1, w2, w3)

tv = 19.2
radio = 35.9
newspaper = 51.3
X , y = prepare_data ('advertising.csv')
( w1 , w2 , w3 ,b , losses ) = implement_linear_regression(X , y , epoch_max =50 , lr =1e-5)
sales = predict ( tv , radio , newspaper , w1 , w2 , w3 , b )
print (f' predicted sales is { sales }')

def implement_linear_regression_nsamples(X_data, y_data, epoch_max = 50, lr= 1e-5):
    losses = []

    w1, w2, w3, b = initialize_params()

    N = len(y_data)

    for epoch in range (epoch_max):

        loss_total = 0
        dw1_total = 0
        dw2_total = 0
        dw3_total = 0
        db_total = 0

        for i in range (N):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]

            y = y_data[i]

            #compute output
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            #compute loss
            loss = compute_loss_mse(y, y_hat)

            #accumulate loss
            loss_total += loss

            #compute gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            #accumulate gradient w1, w2, w3,b
            dw1_total += dl_dw1
            dw2_total += dl_dw2
            dw3_total += dl_dw3
            db_total += dl_db

        #average derivative loss
        dw1_total = dw1_total / N
        dw2_total = dw2_total / N
        dw3_total = dw3_total / N
        db_total = db_total / N

        #update parameters
        w1 = update_weight_wi( w1 , dw1_total , lr )
        w2 = update_weight_wi( w2 , dw2_total , lr )
        w3 = update_weight_wi( w3 , dw3_total , lr )
        b = update_weight_b( b , db_total , lr )

        #logging
        losses.append(loss_total)

    return w1, w2, w3, b, losses

(w1, w2, w3, b, losses) = implement_linear_regression_nsamples(X, y, epoch_max=1000, lr =1e-5)
print(losses)
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

(w1, w2, w3, b, losses) = implement_linear_regression_nsamples(X, y, epoch_max=1000, lr =1e-5)

print ( w1 , w2 , w3 )

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()

    #get tv
    tv_data = get_column(data,0)

    #get radio
    radio_data = get_column(data,1)

    #get newspaper
    newspaper_data = get_column(data,2)

    #get sales
    sales_data = get_column(data,3)
    #Create list of features for input
    X = np.array([[1 , x1 , x2 , x3] for x1 , x2 , x3 in zip( tv_data , radio_data , newspaper_data )])
    y = sales_data
    return X, y

def initialize_params():
    return [0 , -0.01268850433497871 , 0.004752496982185252 , 0.0073796171538643845]

def predict(X_features, weights):
    X_features = np.array(X_features)
    weights = np.array(weights)
    print(X_features.shape)
    print(weights.shape)
    y_hat = X_features @ weights.T
    return y_hat

weights = initialize_params()
X, y = prepare_data('advertising.csv')
y_hat = predict(X, weights)
print(y_hat.shape)

def compute_loss_mse(y, y_hat):
    loss = np.float64((y - y_hat)**2)
    return loss

def compute_gradient_w(X_features, y, y_hat):
    diff = np.array([y_hat - y])
    dl_dw = 2 * diff @ X_features.reshape(1, -1)  # Reshape X_features to (1, n_features)
    return dl_dw    

def update_weight(weights, dl_weights, lr):
    weights = weights - lr * dl_weights
    return weights

def implement_linear_regression(X_features, y_output, epoch_max = 50, lr = 1e-5):
    losses = []
    weights = initialize_params()
    N = len(y_output)
    for epoch in range(epoch_max):
        print(f'epoch {epoch}')
        for i in range(N):
            feature_i = X_features[i]
            y = y_output[i]
            y_hat = predict(feature_i, weights)
            loss = compute_loss_mse(y, y_hat)

            #compute gradient
            dl_dweights = compute_gradient_w(feature_i, y, y_hat)
            #update weights
            weights = update_weight(weights, dl_dweights, lr)
            losses.append(loss)
    return weights, losses

X , y = prepare_data ('advertising.csv')
W , L = implement_linear_regression(X , y , epoch_max =50 , lr =1e-5)
# Print loss value at iteration 9999
print (L[9999])