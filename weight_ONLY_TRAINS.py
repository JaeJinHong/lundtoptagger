import argparse
from datetime import datetime

import torch
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils_newdata import *

print("Libraries loaded!")


def main():

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)

    path_to_file = config['data']['path_to_trainfiles']

    print("dataset used:", path_to_file)

    dataset = torch.load(path_to_file)

    # check the number of signal and background jets
    labels = [data.y for data in dataset]
    num_signal = labels.count(1)
    num_background = labels.count(0)
    print("Signal count:", num_signal)
    print("Background count:", num_background)

    ## define architecture
    batch_size = config['architecture']['batch_size']
    test_size = config['architecture']['test_size']

    dataset= shuffle(dataset,random_state=42)
    train_ds, validation_ds = train_test_split(dataset, test_size = test_size, random_state = 144)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)


    print ("train dataset size:", len(train_ds))
    print ("validation dataset size:", len(validation_ds))

    deg = torch.zeros(100, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())


    n_epochs = config['architecture']['n_epochs']
    learning_rate = config['architecture']['learning_rate']
    choose_model = config['architecture']['choose_model']
    save_every_epoch = config['architecture']['save_every_epoch']

    if choose_model == "LundNet":
        model = LundNet()
    if choose_model == "GATNet":
        model = GATNet()
    if choose_model == "GINNet":
        model = GINNet()
    if choose_model == "EdgeGinNet":
        model = EdgeGinNet()
    if choose_model == "PNANet":
        model = PNANet()

    path_to_ckpt = config['retrain']['path_to_ckpt']

    if config['retrain']['flag']:
        path = path_to_ckpt
        model.load_state_dict(torch.load(path))

    if torch.cuda.is_available():
        device_id = 'cuda' if config['gpu'] is None else 'cuda:'+str(config['gpu'])
    else:
        device_id = 'cpu'
    device = torch.device(device_id)
    print(f'Using device: {device}')

    #model = torch.nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=4*learning_rate)
    optimizer3 = torch.optim.Adam(model.parameters(), lr=10*learning_rate)

    train_jds = []
    val_jds = []

    train_bgrej = []
    val_bgrej = []

    model_name = config['data']['model_name']
    path_to_save = config['data']['path_to_save']
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    metrics_filename = path_to_save+"losses_"+model_name+datetime.now().strftime("%d%m-%H%M")+".txt"

    for epoch in range(n_epochs):
        train_loss.append(train_clas(train_loader, model, device, optimizer, optimizer2, optimizer3, epoch))
        val_loss.append(my_test(val_loader, model, device))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, train_loss[epoch], val_loss[epoch]))
        if (save_every_epoch):
            torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")
        elif epoch == n_epochs-1:
            torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")

    metrics = pd.DataFrame({"Train_Loss":train_loss,"Val_Loss":val_loss})
    metrics.to_csv(metrics_filename, index = False)
    return


if __name__ == "__main__":
    main()

