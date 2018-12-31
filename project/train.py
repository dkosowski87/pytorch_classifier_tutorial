from data.data import training_loader


def train(net, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in training_loader:
            inputs, labels = batch

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(training_loader)
        print('Epoch: ', epoch + 1, ' Loss: ', running_loss)

    print('Finished training.')
    print('Final loss: ', running_loss)
