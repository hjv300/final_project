import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from experiment1b_finetuning import train
# Dataset Loader for Experiment 2
class T5DatasetExp2(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_len=64,
        train_threshold=22,
        test_min_length=24,
        mode='train',
        min_command_length=None,
        max_command_length=None,
        min_action_length=None,
        max_action_length=None,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_threshold = train_threshold
        self.test_min_length = test_min_length
        self.mode = mode

        self.min_command_length = min_command_length
        self.max_command_length = max_command_length
        self.min_action_length = min_action_length
        self.max_action_length = max_action_length

        self.data = []
        with open(data_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                in_text = line.split('IN:')[1].split('OUT:')[0].strip()
                out_text = line.split('OUT:')[1].strip()

                command_length = len(in_text.split())
                action_length = len(out_text.split())

                if self.mode == 'train' and action_length <= self.train_threshold:
                    pass
                elif self.mode == 'test' and action_length >= self.test_min_length:
                    pass
                else:
                    continue

                if (self.min_command_length is not None and command_length < self.min_command_length):
                    continue
                if (self.max_command_length is not None and command_length > self.max_command_length):
                    continue
                if (self.min_action_length is not None and action_length < self.min_action_length):
                    continue
                if (self.max_action_length is not None and action_length > self.max_action_length):
                    continue

                self.data.append((in_text, out_text))

        print(f"{self.mode.capitalize()} set: {len(self.data)} samples loaded "
              f"(cmd_len=[{self.min_command_length},{self.max_command_length}], "
              f"act_len=[{self.min_action_length},{self.max_action_length}]).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_text, out_text = self.data[idx]

        input_encoding = self.tokenizer(
            f"translate English to actions: {in_text}",
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        output_encoding = self.tokenizer(
            out_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': output_encoding['input_ids'].squeeze(),
        }


def plot_acc(sequence_lengths, accuracies, xlabel, ylabel, title, file_name):
    plt.figure(figsize=(8, 6))
    plt.bar(sequence_lengths, accuracies, color='skyblue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate(model, dataloader, tokenizer, device, top_k=1):
    print(f"Evaluate!! Sampling from top {top_k}")
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0
    total_correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=labels.shape[1],
                do_sample=False,
                top_k=top_k
            )

            correct_tokens = (outputs == labels)
            non_pad_mask = (labels != tokenizer.pad_token_id)
            total_correct_tokens += (correct_tokens & non_pad_mask).sum().item()
            total_tokens += non_pad_mask.sum().item()

            correct_sequences = (outputs == labels).all(dim=1)
            total_correct_sequences += correct_sequences.sum().item()
            total_sequences += labels.size(0)

    token_acc = total_correct_tokens / total_tokens if total_tokens != 0 else 0
    sequence_acc = total_correct_sequences / total_sequences if total_sequences != 0 else 0

    print(f"Token-Level Accuracy: {token_acc:.4f}")
    print(f"Sequence-Level Accuracy: {sequence_acc:.4f}")
    return token_acc, sequence_acc

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    path_test = '../data/length_split/tasks_test_length.txt'
    path_train = '../data/length_split/tasks_train_length.txt'
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    #model.load_state_dict(torch.load('t5_experiment2_model.pth'))
    model = train(model, path_train, tokenizer, device, epochs=10, batch_size=16)
    batch_size = 4

    command_sequence_length = [4, 6, 7, 8, 9]
    token_acc_command_statistics = defaultdict(list)
    seq_acc_command_statistics = defaultdict(list)

    for length in command_sequence_length:

        test_dataset = T5DatasetExp2(
            data_path=path_test,
            tokenizer=tokenizer,
            mode='test',
            min_command_length=length,
            max_command_length=length
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        token_acc, seq_acc = evaluate(model, test_loader, tokenizer, device, top_k=1)
        token_acc_command_statistics[length].append(token_acc)
        seq_acc_command_statistics[length].append(seq_acc)

    # Plot for command length
    plot_acc(
        command_sequence_length,
        [token_acc_command_statistics[k][0] for k in command_sequence_length],
        "Command Length",
        "Token-Level Accuracy",
        "Token-Level Accuracy by Command Length",
        "token_acc_command.png"
    )

    plot_acc(
        command_sequence_length,
        [seq_acc_command_statistics[k][0] for k in command_sequence_length],
        "Command Length",
        "Sequence-Level Accuracy",
        "Sequence-Level Accuracy by Command Length",
        "seq_acc_command.png"
    )

    action_sequence_length = [24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48]
    token_acc_action_statistics = defaultdict(list)
    seq_acc_action_statistics = defaultdict(list)

    for length in action_sequence_length:
        test_dataset = T5DatasetExp2(
            data_path=path_test,
            tokenizer=tokenizer,
            mode='test',
            min_action_length=length,
            max_action_length=length
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        token_acc, seq_acc = evaluate(model, test_loader, tokenizer, device, top_k=1)
        token_acc_action_statistics[length].append(token_acc)
        seq_acc_action_statistics[length].append(seq_acc)

    # Plot for action length
    plot_acc(
        action_sequence_length,
        [token_acc_action_statistics[k][0] for k in action_sequence_length],
        "Action Sequence Length",
        "Token-Level Accuracy",
        "Token-Level Accuracy by Action Length",
        "token_acc_action.png"
    )

    plot_acc(
        action_sequence_length,
        [seq_acc_action_statistics[k][0] for k in action_sequence_length],
        "Action Sequence Length",
        "Sequence-Level Accuracy",
        "Sequence-Level Accuracy by Action Length",
        "seq_acc_action.png"
    )

if __name__ == "__main__":
    main()
