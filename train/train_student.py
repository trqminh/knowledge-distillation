import time
import copy
from utils import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


'''
train model:
+ model.train()
+ scheduler.step()
+ iterate data
    ++ out = model(inp)
    ++ optimizer.zero_grad()
    ++ loss 
    ++ loss.backward()
    ++ optimizer.step()

'''


def train_student_model(data_dict, model, _teacher_model, device, alpha, num_epochs=25):
    # load data
    data_loaders = data_dict['data_loaders']
    dataset_sizes = data_dict['dataset_sizes']

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            _teacher_model.eval()

            running_loss = 0.0
            running_corrects = 0
            # checks_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                soft_labels = _teacher_model(inputs).to('cpu').detach().numpy()
                soft_labels = torch.from_numpy(soft_labels).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # _, checks = torch.max(soft_labels, 1)

                    loss1 = criterion(outputs, labels)
                    loss2 = SoftLabelCrossEntropy(outputs, soft_labels, 25)

                    loss = alpha * loss1 + (1 - alpha) * loss2
                    # loss = loss1 + loss2

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                if phase == 'train':
                    running_loss += loss.item() * inputs.size(0)
                else:
                    running_loss += loss1.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
                # checks_corrects += torch.sum(checks == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # epoch_check_acc = checks_corrects.double() / dataset_sizes[phase]

            # print('teacher model prediction accuracy: ', epoch_check_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('**** BEST MODEL IS UPDATED ****')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return best_acc.item()
