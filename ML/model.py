import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

import preprocess as pp

ANNOTATION_SYMBOLS = pp.ANNOTATION_SYMBOLS
NORMALISED_WIDTH = pp.NORMALISED_WIDTH
DEFAULT_PATH = pp.DEFAULT_PATH
RECORDING_FREQUENCY = 32


class DataSet(torch.utils.data.Dataset):
  def __init__(self, x, y):
    super(DataSet, self).__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Raw tensors
    self.x = x.to(self.device)
    self.y = y.to(self.device)

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, index):
    x = self.x[index, :]
    y = self.y[index, :]
    return x, y

"""Fully Convolutional Network for ECG beat classification.

Architecture based on the FCN described in:
  - https://arxiv.org/pdf/1611.06455.pdf (original FCN paper), and
  - https://arxiv.org/pdf/1809.04356.pdf (evaluation)
"""
class FCN(nn.Module):
    def __init__(self, path):
        super(FCN, self).__init__()
        self.path = path
        self.preproc = pp.Preprocessor(self.path)
        
        # One-hot encoding of the annotation symbols
        # E.g. N: [1, 0, 0], A: [0, 1, 0], V: [0, 0, 1]
        self.symbol_mapping = {symbol: i for i, symbol in enumerate(ANNOTATION_SYMBOLS)}
        print("Symbol mapping:", self.symbol_mapping)
        self.encoded_symbols = np.eye(len(ANNOTATION_SYMBOLS))
        
        # Try to use a GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 128, 8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )            

        self.conv_3 = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.reduce_dim = nn.Sequential(
            # Reduce the number of feature maps to the number of classes
            nn.Conv1d(128, len(self.encoded_symbols), 1),
        )

        self.model = nn.Sequential(
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.reduce_dim,
        )
        self.model.apply(self.init_weights)

        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.f1_scores = []

    """Set biases to 0 and perform Xavier-Glorot weight initialisation."""
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
    
    """Forward pass of the model."""
    def forward(self, x):
        # Pass through the convolutional layers
        x1 = self.model(x.to(self.device).unsqueeze(1))

        # Global average pooling to get a feature vector
        x2 = torch.mean(x1.to(self.device), dim=2)
        return x2
    
    """Split the triplets into training, validation, and test sets."""
    def train_val_test_split(self, train_size=0.8, val_size=0.1, test_size=0.1):
        assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"

        # Preprocess the triplets in self.path
        all_triplets, _, _ = self.preproc.preprocess_ecg_records()

        # Group the triplets by the middle annotation (i.e. the one we want to predict)
        grouped_counts = self.preproc.count_by_middle_annotation(all_triplets)
        min_count = min(grouped_counts.values())
        print("Grouped counts:", grouped_counts)

        # Downsample the triplets to balance the classes so that they all have the min_count examples
        downsampled_count = {}
        downsampled_triplets = []
        for triplet in all_triplets:
            key = triplet.annotations[1]
            if downsampled_count.get(key, 0) < min_count:
                downsampled_triplets.append(triplet)
                downsampled_count[key] = downsampled_count.get(key, 0) + 1
    
        print("Downsampled counts:", self.preproc.count_by_middle_annotation(downsampled_triplets))

        # Split the (shuffled) triplets into training, validation, and test sets
        np.random.seed(42)
        np.random.shuffle(downsampled_triplets)
        train_size = int(train_size * len(downsampled_triplets))
        val_size = int(val_size * len(downsampled_triplets))
        train_triplets = np.array(downsampled_triplets[:train_size])
        val_triplets = np.array(downsampled_triplets[train_size:train_size+val_size])
        test_triplets = np.array(downsampled_triplets[train_size+val_size:])

        print("Train counts:", self.preproc.count_by_middle_annotation(train_triplets))
        print("Val counts:", self.preproc.count_by_middle_annotation(val_triplets))
        print("Test counts:", self.preproc.count_by_middle_annotation(test_triplets))

        train_x = torch.tensor(np.array([
            triplet.signal for triplet in train_triplets
        ]), dtype=torch.float32)

        train_y = torch.tensor(np.array([
            self.encoded_symbols[
                self.symbol_mapping[triplet.annotations[1]
            ]] for triplet in train_triplets
        ]), dtype=torch.float32)

        self.train_set = DataSet(train_x, train_y)
        
        val_x = torch.tensor(np.array([
            triplet.signal for triplet in val_triplets
        ]), dtype=torch.float32)

        val_y = torch.tensor(np.array([
            self.encoded_symbols[
                self.symbol_mapping[triplet.annotations[1]
            ]] for triplet in val_triplets
        ]), dtype=torch.float32)

        self.val_set = DataSet(val_x, val_y)

        test_x = torch.tensor(np.array([
            triplet.signal for triplet in test_triplets
        ]), dtype=torch.float32)

        test_y = torch.tensor(np.array([
            self.encoded_symbols[
                self.symbol_mapping[triplet.annotations[1]
            ]] for triplet in test_triplets
        ]), dtype=torch.float32)

        self.test_set = DataSet(test_x, test_y)

        print("Train X shape:", self.train_set.x.shape)
        print("Train Y shape:", self.train_set.y.shape)
        print(self.train_set.x[:0])
        print(self.train_set.y[:0])

    """Save the model to disk.
    
    For loading:
        model = TheModelClass(*args, **kwargs)
        optimizer = TheOptimizerClass(*args, **kwargs)

        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()
        # - or -
        model.train()
    """
    def save_checkpoint(self, state, filename='checkpoint.pth'):
        torch.save(state, filename)

    """Compute a confusion matrix."""
    def confusion_matrix(self, y_true, y_pred):
        confusion_matrix = torch.zeros(len(ANNOTATION_SYMBOLS), len(ANNOTATION_SYMBOLS))
        for i in range(len(y_true)):
            true_class = y_true[i].argmax()
            pred_class = y_pred[i].argmax()
            confusion_matrix[true_class, pred_class] += 1
        return confusion_matrix
    
    """Print a confusion matrix."""
    def print_confusion_matrix(self, confusion_matrix):
        print("Confusion matrix:")
        print("True\Pred".center(9), end="")
        for symbol in ANNOTATION_SYMBOLS:
            print(f"{symbol}".center(9), end="")
        print()
        for i, row in enumerate(confusion_matrix):
            print(ANNOTATION_SYMBOLS[i].center(9), end="")
            for value in row:
                print(f"{int(value.item())}".center(9), end="")
            print()
    
    """Compute precision, recall, F1 score, and accuracy for each class."""
    def compute_metrics(self, confusion_matrix):
        precisions = {}
        recalls = {}
        f1_scores = {}
        for i, symbol in enumerate(ANNOTATION_SYMBOLS):
            tp = confusion_matrix[i, i].item()
            fp = confusion_matrix[:, i].sum().item() - tp
            fn = confusion_matrix[i, :].sum().item() - tp

            # If there are no false positives, precision is 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else (1 if fp == 0 else 0)
            
            # If there are no false negatives, recall is 1
            recall = tp / (tp + fn) if (tp + fn) > 0 else (1 if fn == 0 else 0)
            
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0
            )

            precisions[symbol] = precision
            recalls[symbol] = recall
            f1_scores[symbol] = f1_score

        accuracy = confusion_matrix.diag().sum().item() / confusion_matrix.sum().item()

        return precisions, recalls, f1_scores, accuracy
    
    """Train the model using batched gradient descent.
    
    The best model (based on the validation loss) is saved to disk.
    """
    def train(self, epochs=100, lr=0.001, batch_size=16, batch_generator=None):
        # Measure the time taken to train the model
        start_time = time.time()

        if batch_generator is None:
            batch_generator = torch.Generator().manual_seed(42)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_f1 = 0

        for epoch in range(epochs):
            # Shuffle the training data every epoch to avoid overfitting on patterns in batches
            loader = torch.utils.data.DataLoader(
                self.train_set,
                shuffle=True,
                batch_size=batch_size,
                generator=batch_generator
            )

            sum_train_loss = 0
            train_loss_count = 0
            mini_sum_train_loss = 0

            # Perform minibatched gradient descent
            for i, minibatch in enumerate(loader):
                self.model.train()
                optimizer.zero_grad()

                inputs, targets = minibatch[0], minibatch[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass and loss calculation
                outputs = self.forward(inputs)
                train_loss = self.criterion(outputs, targets)
                sum_train_loss += train_loss.item()
                train_loss_count += 1
                mini_sum_train_loss += train_loss.item()

                # Backward pass and weight update
                train_loss.backward()
                optimizer.step()

                # Record the average training loss and validation loss at regular intervals
                if i % RECORDING_FREQUENCY == 0:
                    self.train_losses.append(mini_sum_train_loss / RECORDING_FREQUENCY)
                    mini_sum_train_loss = 0
                    with torch.no_grad():
                        self.model.eval()
                        val_outputs = self.forward(self.val_set.x)
                        val_loss = self.criterion(val_outputs, self.val_set.y)
                        self.val_losses.append(val_loss.item())

                        # Compute metrics and record the macro-averaged F1 score
                        confusion_matrix = self.confusion_matrix(self.val_set.y, val_outputs)
                        _, _, f1s, _ = self.compute_metrics(confusion_matrix)
                        avg_f1 = sum(f1s.values()) / len(f1s)
                        self.f1_scores.append(avg_f1)

            # train_losses.append(sum_train_loss / len(loader))

            # Evaluate the model on the validation set at the end of each epoch
            with torch.no_grad():
                self.model.eval()
                val_outputs = self.forward(self.val_set.x)
                val_loss = self.criterion(val_outputs, self.val_set.y)
                # val_losses.append(val_loss.item())

                # Compute metrics
                confusion_matrix = self.confusion_matrix(self.val_set.y, val_outputs)
                _, _, f1s, accuracy = self.compute_metrics(confusion_matrix)
                avg_f1 = sum(f1s.values()) / len(f1s)
                # f1_scores.append(avg_f1)

                print(f"Epoch {epoch+1}/{epochs}: avg train loss {sum_train_loss / train_loss_count}, val loss {val_loss.item()} (f1: {avg_f1}, accuracy {accuracy})")
                self.print_confusion_matrix(confusion_matrix)
                # Save the model if it has the best F1-measure so far
                if avg_f1 > best_f1:
                    print("\t> Best model so far (by macro-averaged F1), saving...")
                    best_f1 = avg_f1
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.state_dict(),
                        'best_accuracy': accuracy,
                        'best_f1': avg_f1,
                        'optimizer' : optimizer.state_dict(),
                    }, 'best_model.pth')

        # Save a plot of the training and validation losses on the same graph
        plt.clf()
        plt.plot(self.train_losses, label='Train loss', color='orange')
        plt.plot(self.val_losses, label='Validation loss', color='turquoise')
        plt.xlabel(f"Interval of {RECORDING_FREQUENCY} minibatches")
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('losses.png')

        # Clear the plots and then save a plot of the macro-averaged F1 scores
        plt.clf()
        plt.plot(self.f1_scores, color='mediumorchid')
        plt.xlabel(f"Interval of {RECORDING_FREQUENCY} minibatches")
        plt.ylabel('Macro-averaged F1 (validation set)')
        plt.savefig('f1_scores.png')

    """Evaluate the model on the test set."""
    def test(self):
        self.model.eval()
        test_outputs = self.forward(self.test_set.x)
        test_loss = self.criterion(test_outputs, self.test_set.y)

        # Calculate accuracy (number of correct predictions / total number of predictions)
        correct = (test_outputs.argmax(dim=1) == self.test_set.y.argmax(dim=1)).sum().item()
        accuracy = correct / len(self.test_set.y)

        # Calculate the confusion matrix for each class (row = true class, column = predicted class)
        confusion_matrix = torch.zeros(len(ANNOTATION_SYMBOLS), len(ANNOTATION_SYMBOLS))
        for i in range(len(self.test_set.y)):
            true_class = self.test_set.y[i].argmax()
            pred_class = test_outputs[i].argmax()
            confusion_matrix[true_class, pred_class] += 1

        print(f"Test loss: {test_loss.item()}")
        print(f"Test accuracy: {accuracy}")

        self.print_confusion_matrix(confusion_matrix)
        p, r, f1, a = self.compute_metrics(confusion_matrix)
        print("Precision:", p)
        print("Recall:", r)
        print("F1 score:", f1)
        print("Accuracy:", a)

def main():
    model = FCN(DEFAULT_PATH)
    model.to(model.device)

    model.train_val_test_split()

    # comment this out to not train again
    # start = time.time()
    # model.train(epochs=100)
    # print(f"Training took {time.time() - start:.2f} seconds")

    # Load the best model and evaluate it on the test set
    # No need to load the optimizer state etc. because only evaluating
    # checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))

    # if torch.cuda.is_available:
        # model.load_state_dict(checkpoint['state_dict'])
    # else:
        # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()

    # model.test()
main()