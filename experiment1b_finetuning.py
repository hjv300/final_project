import random
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, Subset
import torch

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Dataset with Subsampling Support
class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_len=64):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        with open(data_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                in_text = line.split('IN:')[1].split('OUT:')[0].strip()
                out_text = line.split('OUT:')[1].strip()
                self.data.append((in_text, out_text))

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

# 2. Training and Evaluation Functions
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    total_correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, label in zip(decoded_preds, decoded_labels):
                if pred.strip() == label.strip():
                    total_correct_sequences += 1
                total_sequences += 1

    return total_correct_sequences / total_sequences

# 3. Automate Training Across Data Percentages
def run_experiments(data_path, percentages, epochs=10, batch_size=16):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    full_dataset = T5Dataset(data_path, tokenizer)

    results = {}

    for pct in percentages:
        subset_size = int(len(full_dataset) * pct)
        subset_indices = random.sample(range(len(full_dataset)), subset_size)
        subset = Subset(full_dataset, subset_indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
        optimizer = AdamW(model.parameters(), lr=3e-4)

        print(f"\n--- Training on {int(pct * 100)}% of the data ({subset_size} samples) ---")
        for epoch in range(epochs):
            train_loss = train(model, dataloader, optimizer, device)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")

        # Evaluate the model
        eval_loader = DataLoader(full_dataset, batch_size=batch_size)
        accuracy = evaluate(model, eval_loader, tokenizer, device)
        print(f"Final Sequence Accuracy for {int(pct * 100)}% data: {accuracy:.4f}")

        # Save results
        results[f"{int(pct * 100)}%"] = accuracy

    return results

# 4. Run the Experiments
percentages = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
results = run_experiments('../data/simple_split/tasks_train_simple.txt', percentages)

# 5. Display Results
print("\n--- Final Results ---")
for pct, acc in results.items():
    print(f"Data Used: {pct} | Sequence Accuracy: {acc:.4f}")
