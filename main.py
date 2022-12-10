import os, sys, pdb, math
import numpy as np
import random
import torch
from torch import nn

from tqdm import tqdm

from utils import set_seed, setup_gpus, check_directories
from dataloader import load_data, load_tokenizer, get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from arguments import params
import transformers

from sparsemodel import SparseModel

def train(args, model, datasets, val=True):
    criterion = nn.CrossEntropyLoss()

    dataloader = get_dataloader(args, datasets["train"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(1, args.n_epochs + 1):
        model.train()

        avg_loss = 0
        avg_acc = 0

        # evolve the model
        if args.evolution_epochs == -1 or epoch <= args.evolution_epochs:
            model.evolve()

        for batch in tqdm(dataloader):
            inputs, labels = prepare_inputs(args, batch)
            predictions = model(**inputs)

            loss = criterion(predictions, labels)
            avg_loss += loss.item() / len(dataloader)
            avg_acc += (predictions.argmax(1) == labels).float().sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.freeze_zero_weights()

        avg_acc /= len(datasets["train"])
        print('epoch', epoch, '| loss:', avg_loss, "acc:", avg_acc)

        # collect stats for the run
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        if val:
            val_loss, val_acc = run_eval(args, model, datasets, criterion)
            val_losses.append(val_loss)
            val_accs.append(val_acc)


    return train_losses, train_accs, val_losses, val_accs


def run_eval(args, model, datasets, criterion, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    with torch.no_grad():
        avg_loss = 0
        avg_acc = 0
        for batch in tqdm(dataloader):
            inputs, labels = prepare_inputs(args, batch)
            predictions = model(**inputs)

            loss = criterion(predictions, labels)
            avg_loss += loss.item() / len(dataloader)
            avg_acc += (predictions.argmax(1) == labels).float().sum().item()

        avg_acc /= len(datasets[split])
        print(f'{split} loss: {avg_loss}, acc: {avg_acc}')
        return avg_loss, avg_acc

def grid_search(args, datasets, tokenizer):
    import itertools

    ranges = (6,6,6,6)
    acc_matrix = np.zeros(ranges)

    in_pweights = np.linspace(0.05, 0.95, ranges[0])
    zetas = np.linspace(0.05, 0.5, ranges[1])
    out_pweights = np.linspace(0.05, 0.95, ranges[2])
    out_zetas = np.linspace(0.05, 0.5, ranges[3])
    for i, params in enumerate(itertools.product(in_pweights, zetas, out_pweights, out_zetas)):
        args.sparse_layer_pweight = params[0]
        args.sparse_layer_zeta = params[1]
        args.classifier_pweight = params[2]
        args.classifier_zeta = params[3]

        model = SparseModel(args, tokenizer, target_size=60).to(args.device)

        index = np.unravel_index(i, acc_matrix.shape)
        train(args, model, datasets, val=False)
        loss, acc = run_eval(args, model, datasets, criterion, split="test")
        acc_matrix[index] = acc
        np.save(args.grid_search_output, acc_matrix)

        #with open(args.grid_search_output, 'w') as f:
        #    print(acc_matrix, file=f)
        #    print(acc_matrix.flatten(), file=f)

def save_stats(stats, name):
    train_loss, train_acc, val_loss, val_acc = stats

    np.save(name + "_train_loss.npy", train_loss)
    np.save(name + "_train_acc.npy", train_acc)
    np.save(name + "_val_loss.npy", val_loss)
    np.save(name + "_val_acc.npy", val_acc)


def get_training_data(args, tokenizer, datasets):
    model = SparseModel(args, tokenizer, target_size=60).to(args.device)
    sparse_stats = train(args, model, datasets)
    run_eval(args, model, datasets, criterion=nn.CrossEntropyLoss(), split="test")

    set_seed(args)
    args.transformer = "dense"
    args.classifier = "dense"
    model = SparseModel(args, tokenizer, target_size=60).to(args.device)
    dense_stats = train(args, model, datasets)
    run_eval(args, model, datasets, criterion=nn.CrossEntropyLoss(), split="test")

    save_stats(sparse_stats, "sparse")
    save_stats(dense_stats, "dense")


if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    set_seed(args)

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(features, tokenizer)

    #grid_search(args, datasets, tokenizer)
    #get_training_data(args, datasets, tokenizer)

    model = SparseModel(args, tokenizer, target_size=60).to(args.device)
    sparse_stats = train(args, model, datasets)
    run_eval(args, model, datasets, criterion=nn.CrossEntropyLoss(), split="test")
