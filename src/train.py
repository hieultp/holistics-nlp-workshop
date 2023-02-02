import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange


def test(model, test_loader, n_test_samples, device, is_regression=False):
    model.eval()
    total_correct = 0
    with torch.inference_mode():
        with tqdm(test_loader, leave=False, mininterval=0.5) as epoch_progress:
            for data, label in epoch_progress:
                data = data.to(device)
                label = label.to(device)

                prediction = model(data)

                # Calculate accuracy
                if is_regression:
                    prediction = prediction.round().squeeze()
                else:
                    prediction = prediction.argmax(dim=1)
                total_correct += (prediction == label).sum().item()

    return total_correct / n_test_samples


def train(model, loss_fn, train_set, test_set, num_epochs=10, collate_fn=None):
    n_test_samples = len(test_set)
    batch_size = 64
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    is_regression = not isinstance(loss_fn, nn.CrossEntropyLoss)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(model.parameters(), lr=1e-4)
    model = model.to(device)
    best_accuracy = 0.0
    best_train_accuracy = 0.0
    with trange(num_epochs) as train_progress:
        for epoch in train_progress:
            model.train()
            with tqdm(
                train_loader, desc=f"Epoch {epoch}", leave=False, mininterval=0.5
            ) as epoch_progress:
                for data, label in epoch_progress:
                    data = data.to(device)
                    label = label.to(device)
                    label = label if not is_regression else label.float()

                    optimizer.zero_grad()
                    prediction = model(data)
                    loss = loss_fn(prediction.squeeze(), label)
                    loss.backward()
                    optimizer.step()

                    # Calculate accuracy
                    if is_regression:
                        prediction = prediction.round().squeeze()
                    else:
                        prediction = prediction.argmax(dim=1)
                    correct = (prediction == label).sum().item()
                    accuracy = correct / batch_size
                    best_train_accuracy = (
                        accuracy
                        if accuracy > best_train_accuracy
                        else best_train_accuracy
                    )
                    epoch_progress.set_postfix(loss=loss.item(), accuracy=accuracy)

            test_accuracy = test(
                model, test_loader, n_test_samples, device, is_regression
            )
            best_accuracy = (
                test_accuracy if test_accuracy > best_accuracy else best_accuracy
            )
            train_progress.set_postfix(
                best_accuracy=best_accuracy,
                best_train_accuracy=best_train_accuracy,
                test_accuracy=test_accuracy,
            )

    return model
