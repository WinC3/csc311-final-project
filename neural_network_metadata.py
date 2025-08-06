import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from utils import (
    load_train_sparse,
    load_valid_csv,
    load_public_test_csv,
    load_train_csv,
    load_question_meta
)

def load_data(base_path="./data"):
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoderWithMetadata(nn.Module):
    def __init__(self, num_question, metadata_dim, k=100):
        super(AutoEncoderWithMetadata, self).__init__()
        self.g = nn.Linear(num_question, k)
        self.meta_layer = nn.Linear(metadata_dim, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        return (
            torch.norm(self.g.weight, 2) ** 2 +
            torch.norm(self.meta_layer.weight, 2) ** 2 +
            torch.norm(self.h.weight, 2) ** 2
        )

    def forward(self, inputs, question_metadata_matrix=None):
        user_latent = torch.sigmoid(self.g(inputs))  # shape: [1, k]

        if question_metadata_matrix is not None:
            meta_vector = torch.mean(question_metadata_matrix.float(), dim=0, keepdim=True)
            meta_latent = torch.sigmoid(self.meta_layer(meta_vector))  # shape: [1, k]
            latent = user_latent + meta_latent
        else:
            latent = user_latent

        out = torch.sigmoid(self.h(latent))  # shape: [1, num_question]
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, question_meta_tensor, num_epoch):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epoch):
        train_loss = 0.0
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs, question_meta_tensor)

            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0)
            loss += (lamb / 2) * model.get_weight_norm()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        valid_acc = evaluate(model, zero_train_data, valid_data, question_meta_tensor)
        train_losses.append(train_loss)
        val_accuracies.append(valid_acc)

        print(f"Epoch {epoch} \t Train Loss: {train_loss:.4f} \t Valid Acc: {valid_acc:.4f}")

    return train_losses, val_accuracies


def evaluate(model, train_data, eval_data, question_meta_tensor):
    model.eval()
    correct = 0
    total = 0

    for i, u in enumerate(eval_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs, question_meta_tensor)
        pred = output[0][eval_data["question_id"][i]].item() >= 0.5
        if pred == eval_data["is_correct"][i]:
            correct += 1
        total += 1

    return correct / float(total)


def prepare_question_metadata_tensor(question_meta, num_questions, subject_vocab):
    meta_tensor = torch.zeros(num_questions, len(subject_vocab))
    for qid_str, subject_ids in question_meta.items():
        qid = int(qid_str)
        for sid in subject_ids:
            if sid in subject_vocab:
                idx = subject_vocab[sid]
                meta_tensor[qid][idx] = 1.0
    return meta_tensor


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Load metadata
    raw_question_meta = load_question_meta("./data/question_meta.csv")
    all_subjects = sorted({sid for q in raw_question_meta.values() for sid in q})
    subject_vocab = {sid: i for i, sid in enumerate(all_subjects)}
    question_meta_tensor = prepare_question_metadata_tensor(
        raw_question_meta, train_matrix.shape[1], subject_vocab
    )

    metadata_dim = len(subject_vocab)
    model = AutoEncoderWithMetadata(num_question=train_matrix.shape[1],
                                     metadata_dim=metadata_dim,
                                     k=10)

    lr = 0.03
    lamb = 0.0001
    num_epoch = 30

    train_losses, val_accuracies = train(
        model, lr, lamb,
        train_matrix, zero_train_matrix,
        valid_data, question_meta_tensor,
        num_epoch
    )

    test_acc = evaluate(model, zero_train_matrix, test_data, question_meta_tensor)
    print("Final Test Accuracy:", test_acc)

    # Plot training loss and validation accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(range(num_epoch), train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.legend()

    ax2.plot(range(num_epoch), val_accuracies, label='Validation Accuracy', color='green')
    ax2.axhline(y=test_acc, linestyle='--', color='red', label='Final Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy over Epochs')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("nn_with_metadata_training_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
