# """
# Created on Tue Jan  2 14:18:41 2024
# @author: HIGHer
# """

# import os
# import io
# import pickle
# import torch
# from torch.utils.data import TensorDataset, DataLoader, random_split
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from torchvision import datasets, transforms

# from simple_cnn_config import SimpleCNN  # Make sure this exists and defines your model

# # --- Global path setup ---
# script_dir = os.path.dirname(os.path.abspath(__file__))
# main_dir = os.path.dirname(script_dir)

# # ============================ #
# #     Dataset Preprocessors   #
# # ============================ #

# def preprocess_uci_har(dataset_addr):
#     x_train_file = os.path.normpath(os.path.join(dataset_addr, 'X_train.txt'))
#     y_train_file = os.path.normpath(os.path.join(dataset_addr, 'y_train.txt'))

#     print(f"DEBUG: Loading X_train from: {x_train_file}")
    
#     x_train = np.loadtxt(x_train_file)
#     y_train = np.loadtxt(y_train_file)

#     label_encoder = LabelEncoder()
#     y_train_encoded = label_encoder.fit_transform(y_train)

#     x_tensor = torch.FloatTensor(x_train)
#     y_tensor = torch.LongTensor(y_train_encoded)

#     dataset = TensorDataset(x_tensor, y_tensor)
#     input_size = x_train.shape[1]
#     return dataset, input_size


# def preprocess_mnist(dataset_addr):
#     transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     mnist_dataset = datasets.MNIST(root=dataset_addr, train=True, download=True, transform=transform)
#     input_size = 28
#     return mnist_dataset, input_size


# # ============================ #
# #         Train Model         #
# # ============================ #

# def train_model(model, dataloader, validation_dataloader, criterion, optimizer, epochs, device):
#     print_every = len(dataloader)
#     model.to(device)

#     for epoch in range(epochs):
#         running_loss = 0.0
#         model.train()
#         for i, (inputs, labels) in enumerate(dataloader, 1):
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             if i % print_every == 0:
#                 print(f'Epoch {epoch+1}, Batch {i}/{len(dataloader)}, Loss: {running_loss/print_every:.4f}')
#                 running_loss = 0.0

#         validate_model(model, validation_dataloader, device)


# def validate_model(model, dataloader, device):
#     model.eval()
#     total_correct = 0
#     total_samples = 0

#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total_samples += labels.size(0)
#             total_correct += (predicted == labels).sum().item()

#     accuracy = total_correct / total_samples
#     print(f'Validation Accuracy: {accuracy:.4f}')


# # ============================ #
# #      Main Train Wrapper     #
# # ============================ #

# # def train(global_model_bytes, num_epochs, dataset_type):
# #     """
# #     Train a local model using weights from a global model.
# #     Accepts serialized weights from torch.save(...) or pickle.dumps(...).

# #     Returns:
# #         model (torch.nn.Module): Trained local model
# #     """

# #     # --- Prepare dataset ---
# #     if dataset_type == "UCI_HAR":
# #         dataset_addr = os.path.join(main_dir, 'dataset', 'UCI_HAR_Dataset', 'train')
# #         dataset, input_size = preprocess_uci_har(dataset_addr)
# #     elif dataset_type == "MNIST":
# #         dataset_addr = os.path.join(main_dir, 'dataset')
# #         dataset, input_size = preprocess_mnist(dataset_addr)
# #     else:
# #         raise ValueError("Unsupported dataset_type")

# #     torch.manual_seed(42)

# #     if isinstance(dataset, TensorDataset):
# #         data, targets = dataset.tensors
# #         shuffled_indices = torch.randperm(len(data))
# #         shuffled_dataset = TensorDataset(data[shuffled_indices], targets[shuffled_indices])
# #     else:
# #         shuffled_dataset = dataset

# #     train_size = int(0.8 * len(shuffled_dataset))
# #     validation_size = len(shuffled_dataset) - train_size
# #     train_dataset, validation_dataset = random_split(shuffled_dataset, [train_size, validation_size])

# #     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# #     validation_dataloader = DataLoader(validation_dataset, batch_size=64)

# #     # --- Build and Load Model ---
# #     model = SimpleCNN(dataset_type=dataset_type)
# #     loaded_state = None

# #     # Try to load via torch.load
# #     try:
# #         buffer = io.BytesIO(global_model_bytes)
# #         loaded = torch.load(buffer, map_location='cpu')

# #         if isinstance(loaded, dict):
# #             loaded_state = loaded
# #             print("DEBUG: Loaded global model via torch.load (state_dict).")
# #         elif hasattr(loaded, 'state_dict'):
# #             loaded_state = loaded.state_dict()
# #             print("DEBUG: Loaded full model via torch.load, using state_dict.")
# #     except Exception as e:
# #         print(f"DEBUG: torch.load failed: {e}. Trying pickle...")

# #     # Fallback: pickle.loads
# #     if loaded_state is None:
# #         try:
# #             maybe_state = pickle.loads(global_model_bytes)
# #             if isinstance(maybe_state, dict):
# #                 loaded_state = maybe_state
# #                 print("DEBUG: Loaded global model via pickle.loads (state_dict).")
# #             else:
# #                 raise TypeError("pickle.loads returned non-dict")
# #         except Exception as e:
# #             print(f"ERROR: Failed to load global model: {e}")
# #             raise

# #     # Load weights into model
# #     try:
# #         model.load_state_dict(loaded_state)
# #         print("DEBUG: state_dict loaded into model.")
# #     except Exception as e:
# #         print(f"ERROR: Failed to apply state_dict to model: {e}")
# #         raise

# #     # --- Training ---
# #     criterion = torch.nn.CrossEntropyLoss()
# #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# #     train_model(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs, device='cpu')

# #     return model


# def train(global_model, num_epochs, dataset_type):
#     # --- Detect input type ---
#     if isinstance(global_model, (bytes, bytearray)):
#         try:
#             maybe_state = pickle.loads(global_model)
#             if isinstance(maybe_state, dict):  # state_dict
#                 model = SimpleCNN(dataset_type=dataset_type)
#                 model.load_state_dict(maybe_state)
#                 print("DEBUG: Loaded model from state_dict (bytes).")
#             else:
#                 model = maybe_state
#                 print("DEBUG: Loaded full model object (bytes).")
#         except Exception as e:
#             print(f"ERROR: Could not unpickle global_model bytes: {e}")
#             raise
#     elif isinstance(global_model, torch.nn.Module):
#         model = global_model
#         print("DEBUG: Using provided model object directly.")
#     else:
#         raise TypeError(f"Unsupported global_model type: {type(global_model)}")

#     # --- Dataset setup ---
#     if dataset_type == "UCI_HAR":
#         dataset_addr = os.path.join(main_dir, 'dataset', 'UCI_HAR_Dataset', 'train')
#         dataset, input_size = preprocess_uci_har(dataset_addr)

#     elif dataset_type == "MNIST":
#         dataset_addr = main_dir + '/dataset/'
#         dataset, input_size = preprocess_mnist(dataset_addr)
#     else:
#         raise ValueError("Something wrong with dataset type")

#     torch.manual_seed(42)

#     # Shuffle dataset
#     if isinstance(dataset, TensorDataset):
#         data, targets = dataset.tensors
#         shuffled_indices = torch.randperm(len(data))
#         shuffled_dataset = TensorDataset(data[shuffled_indices], targets[shuffled_indices])
#     else:
#         shuffled_dataset = dataset

#     train_size = int(0.8 * len(shuffled_dataset))
#     validation_size = len(shuffled_dataset) - train_size
#     train_dataset, validation_dataset = random_split(shuffled_dataset, [train_size, validation_size])

#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     validation_dataloader = DataLoader(validation_dataset, batch_size=64)

#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # --- Train ---
#     train_model(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs, device='cpu')

#     return model







import os
import io
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets, transforms
import torch.optim.lr_scheduler # Added for learning rate scheduler

from simple_cnn_config import SimpleCNN  # Ensure this file exists

# --- Global path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(script_dir)

# ============================ #
#     Dataset Preprocessors    #
# ============================ #

def preprocess_uci_har(dataset_addr):
    x_train_file = os.path.normpath(os.path.join(dataset_addr, 'X_train.txt'))
    y_train_file = os.path.normpath(os.path.join(dataset_addr, 'y_train.txt'))

    print(f"DEBUG: Loading X_train from: {x_train_file}")

    x_train = np.loadtxt(x_train_file)
    y_train = np.loadtxt(y_train_file)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    x_tensor = torch.FloatTensor(x_train)
    y_tensor = torch.LongTensor(y_train_encoded)

    dataset = TensorDataset(x_tensor, y_tensor)
    input_size = x_train.shape[1]
    return dataset, input_size


def preprocess_mnist(dataset_addr):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_dataset = datasets.MNIST(root=dataset_addr, train=True, download=True, transform=transform)
    input_size = 28
    return mnist_dataset, input_size


# ============================ #
#      FedProx Helper          #
# ============================ #

def _state_dict_to_tensor_list(state_dict, device):
    """Convert state_dict tensors to a list of tensors on device."""
    return [torch.tensor(v).to(device) for v in state_dict.values()]


# ============================ #
#         Validate Model       #
# ============================ #

def validate_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f'Validation Accuracy: {accuracy:.4f}')
    return accuracy


# ============================ #
#         Train Model          #
# ============================ #

def train_model(
        model,
        dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs,
        device,
        scheduler=None, # Added scheduler parameter
        global_state_dict=None,
        mu=0.0
    ):
    """
    Train local model. If global_state_dict is provided and mu>0:
       loss' = loss + (mu/2) * Σ ||w - w_global||^2
    """

    model.to(device)

    # Prepare list of tensors representing global weights (FedProx)
    prox_tensors = None
    if global_state_dict is not None and mu > 0:
        prox_tensors = _state_dict_to_tensor_list(global_state_dict, device)

    print_every = len(dataloader)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()

        for i, (inputs, labels) in enumerate(dataloader, 1):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # ---------- FedProx proximal regularizer ----------
            if prox_tensors is not None:
                prox_term = 0.0
                for param, g_w in zip(model.parameters(), prox_tensors):
                    prox_term += torch.sum((param - g_w) ** 2)
                loss = loss + (mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % print_every == 0:
                print(f"[Epoch {epoch+1}] Loss: {running_loss/print_every:.4f}")
                running_loss = 0.0

        validate_model(model, validation_dataloader, device)

        if scheduler is not None:
            scheduler.step() # Update learning rate at the end of each epoch

    return model


# ============================ #
#      Training Wrapper        #
# ============================ #

def train(global_model, num_epochs, dataset_type, mu=0.0):
    """
    Train local model with optional FedProx (mu > 0).
    global_model can be a state_dict(), a full torch.nn.Module, or bytes.
    """

    # --- Detect input type for global model ---
    if isinstance(global_model, (bytes, bytearray)):
        try:
            maybe = pickle.loads(global_model)
            if isinstance(maybe, dict):
                model = SimpleCNN(dataset_type=dataset_type)
                model.load_state_dict(maybe)
                global_state_dict = maybe
                print("DEBUG: Loaded model state_dict from bytes.")
            else:
                model = maybe
                global_state_dict = model.state_dict()
                print("DEBUG: Loaded full model object from bytes.")
        except Exception as e:
            print(f"ERROR: Could not unpickle global model bytes: {e}")
            raise

    elif isinstance(global_model, torch.nn.Module):
        model = global_model
        global_state_dict = model.state_dict()
        print("DEBUG: Using provided torch.nn.Module directly.")

    elif isinstance(global_model, dict):
        model = SimpleCNN(dataset_type=dataset_type)
        model.load_state_dict(global_model)
        global_state_dict = global_model
        print("DEBUG: Using provided state_dict.")

    else:
        raise TypeError(f"Unsupported global_model type: {type(global_model)}")

    # --- Dataset setup ---
    if dataset_type == "UCI_HAR":
        dataset_addr = os.path.join(main_dir, 'dataset', 'UCI_HAR_Dataset', 'train')
        dataset, input_size = preprocess_uci_har(dataset_addr)

    elif dataset_type == "MNIST":
        dataset_addr = os.path.join(main_dir, 'dataset')
        dataset, input_size = preprocess_mnist(dataset_addr)

    else:
        raise ValueError("Invalid dataset_type")

    torch.manual_seed(42)

    # Shuffle dataset
    if isinstance(dataset, TensorDataset):
        data, targets = dataset.tensors
        shuffled_indices = torch.randperm(len(data))
        dataset = TensorDataset(data[shuffled_indices], targets[shuffled_indices])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Added L2 regularization
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # Initialize StepLR

    # --- Train (with FedProx support) ---
    train_model(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        num_epochs,
        device='cpu',
        scheduler=scheduler, # Pass scheduler to train_model
        global_state_dict=global_state_dict,
        mu=mu  # FedProx coefficient (0 = FedAvg)
    )

    return model
