import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


def test(model, test_loader, n_test_samples, device, is_regression=False):
    model.eval()
    total_correct = 0
    with torch.inference_mode():
        with tqdm(test_loader, total=n_test_samples, leave=False) as epoch_progress:
            for img, label in epoch_progress:
                img = img.to(device)
                label = label.to(device)

                prediction = model(img)

                # Calculate accuracy
                if is_regression:
                    prediction = prediction.clip(max=10).round().squeeze()
                else:
                    prediction = prediction.argmax(dim=1, keepdim=True).squeeze()
                total_correct += (prediction == label).sum().item()

    return total_correct / n_test_samples


def train(
    model,
    loss_fn,
    train_set,
    test_set,
    use_gpu=False,
    num_epochs=10,
):
    n_train_samples = len(train_set)
    n_test_samples = len(test_set)
    batch_size = 64
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if use_gpu else False,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if use_gpu else False,
        drop_last=False,
    )

    is_regression = not isinstance(loss_fn, nn.CrossEntropyLoss)
    device = torch.device("cuda" if use_gpu else "cpu")
    optimizer = Adam(model.parameters(), lr=1e-4)
    model = model.to(device)
    with trange(num_epochs) as train_progress:
        for epoch in train_progress:
            model.train()
            with tqdm(
                train_loader, desc=f"Epoch {epoch}", total=n_train_samples, leave=False
            ) as epoch_progress:
                for img, label in epoch_progress:
                    img = img.to(device)
                    label = label.to(device)
                    label = label if not is_regression else label.float()

                    optimizer.zero_grad()
                    prediction = model(img)
                    loss = loss_fn(prediction.squeeze(), label)
                    loss.backward()
                    optimizer.step()

                    # Calculate accuracy
                    if is_regression:
                        prediction = prediction.clip(max=10).round().squeeze()
                    else:
                        prediction = prediction.argmax(dim=1, keepdim=True).squeeze()
                    correct = (prediction == label).sum().item()
                    accuracy = correct / batch_size

                    epoch_progress.set_postfix(loss=loss.item(), accuracy=accuracy)

            test_accuracy = test(
                model, test_loader, n_test_samples, device, is_regression
            )
            train_progress.set_postfix(test_accuracy=test_accuracy)

    return model
