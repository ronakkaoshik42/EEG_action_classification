from xarray import corr
from data_loader import EEGDataset
import torch
from model import CNN
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary

if __name__ == '__main__':
    
    train_data = EEGDataset(train=True,subject_ids=[1])
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32,shuffle=True, num_workers=2)
    test_data = EEGDataset(train=False,subject_ids=[1])
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, num_workers=2)


    net = CNN()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    epochs = 50
    for epoch in range(epochs): 

        running_loss = 0.0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            y_pred = torch.max(outputs,dim=1)[1]
            correct+=torch.sum(y_pred==labels)

        correct_test = 0
        for i, t_data in enumerate(testloader, 0):
            test_inputs, test_labels = t_data[0].to(device), t_data[1].to(device)
            test_outputs = net(test_inputs)
            y_test_pred = torch.max(test_outputs,dim=1)[1]
            correct_test+=torch.sum(y_test_pred==test_labels)
        print('Epoch :',str(epoch+1)+'/'+str(epochs),'Train Acc :',round((correct.item()/len(train_data))*100,2),
              'Test Acc :',round((correct_test.item()/len(test_data))*100,2))

    print('Finished Training')