import numpy as np
# generate sample data

def sample_points(n):
# returns (X,Y), where X of shape (n,2) is the numpy array of points 
# and Y is the (n) array of classes
    radius = np.random.uniform(low=0,high=2,size=n).reshape(-1,1) 
    # uniform radius between 0 and 2
    angle = np.random.uniform(low=0,high=2*np.pi,size=n).reshape(-1,1) 
    # uniform angle
    x1 = radius*np.cos(angle)
    x2=radius*np.sin(angle)
    y = (radius<1).astype(int).reshape(-1)
    x = np.concatenate([x1,x2],axis=1)
    return x,y


# Define training process
def training_routine(net,dataset,n_iters,gpu):
# organize the data
    train_data,train_labels,val_data,val_labels = dataset
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)

# use the flag
#if gpu:
#    train_data,train_labels = train_data.cuda(),train_labels.cuda()
#    val_data,val_labels = val_data.cuda(),val_labels.cuda()
#    net = net.cuda() # the network parameters also need to be on the gpu !
#    print("Using GPU")
#else:
#    print("Using CPU")

for i in range(n_iters):
    # forward pass
    train_output = net(train_data)
    train_loss = criterion(train_output,train_labels)
    # backward pass and optimization
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad() 
    # Once every 100 iterations, print statistics
    if i%100==0:
        print("At iteration",i)
        # compute the accuracy of the prediction
        train_prediction = train_output.cpu().detach().argmax(dim=1)
        train_accuracy = (train_prediction.numpy()==train_labels.numpy()).mean()
        # Now for the validation set
        val_output = net(val_data)
        val_loss = criterion(val_output,val_labels)
        # compute the accuracy of the prediction
        val_prediction = val_output.cpu().detach().argmax(dim=1)
        val_accuracy = (val_prediction.numpy()==val_labels.numpy()).mean()
        print("Training loss :",train_loss.cpu().detach().numpy())
        print("Training accuracy :",train_accuracy)
        print("Validation loss :",val_loss.cpu().detach().numpy())
        print("Validation accuracy :",val_accuracy)

