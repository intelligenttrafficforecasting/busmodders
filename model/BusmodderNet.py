import torch.nn as nn
import numpy as np
class BusmodderNet(nn.Module):
    def train_net(self,train,test):
        # setting hyperparameters and gettings epoch sizes
        batch_size = 12
        num_epochs = 100
        num_samples_train = train.shape[0]-1
        num_batches_train = num_samples_train - batch_size
        num_samples_test = test.shape[0]-1
        num_batches_test = num_samples_test - batch_size 

        # setting up lists for handling loss/accuracy
        train_acc, train_loss = [], []
        valid_acc, valid_loss = [], []
        test_acc, test_loss = [], []
        cur_loss = 0
        losses = []

        get_slice = lambda i, size: range(i - size, i)

        for epoch in range(num_epochs):
            # Forward -> Backprob -> Update params
            ## Train
            cur_loss = 0
            self.train()
            for i in range(batch_size,num_batches_train):
                slce = get_slice(i, batch_size)
                x_batch = Variable(torch.from_numpy(train.values[slce,:]).float().transpose(0, 1))
                output = self(x_batch)

                # compute gradients given loss
                target_batch = Variable(torch.from_numpy(train.values[i+1,:]).long())
                batch_loss = criterion(output, target_batch)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                cur_loss += batch_loss   
            losses.append(cur_loss / batch_size)

            self.eval()
            ### Evaluate training
            train_preds, train_targs = [], []
            for i in range(num_batches_train):
                slce = get_slice(i, batch_size)
                x_batch = Variable(torch.from_numpy(train.values[slce,:]).float().transpose(0, 1))

                output = self(x_batch)
                preds = torch.max(output, 1)[1]

                train_targs += list(train.values[i+1,:])
                train_preds += list(preds.data.numpy())

            ### Evaluate validation
            val_preds, val_targs = [], []
            for i in range(batch_size,num_batches_test):
                slce = get_slice(i, batch_size)
                x_batch = Variable(torch.from_numpy(test.values[slce,:]).float().transpose(0, 1))

                output = self(x_batch)
                preds = torch.max(output, 1)[1]
                val_preds += list(preds.data.numpy())
                val_targs += list(test.values[i+1,:])

            #train_acc_cur = accuracy_score(train_targs, train_preds)
            #valid_acc_cur = accuracy_score(val_targs, val_preds)

            train_acc_cur = sum(train_targs)/len(train_targs)-sum(train_preds)/len(train_preds)
            valid_acc_cur = sum(val_targs)/len(val_targs)-sum(val_preds)/len(val_preds)

            train_acc.append(train_acc_cur)
            valid_acc.append(valid_acc_cur)

            if epoch % 10 == 0:
                print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                        epoch+1, losses[-1], train_acc_cur, valid_acc_cur))

#epoch = np.arange(len(train_acc))
#plt.figure()
#plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
#plt.legend(['Train Accucary','Validation Accuracy'])
#plt.xlabel('Updates'), plt.ylabel('Acc')