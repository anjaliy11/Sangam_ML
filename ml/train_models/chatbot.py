import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForQuestionAnswering
from sklearn.model_selection import train_test_split

# Load the dataset
with open("data/enhanced_data.json", "r") as file:
    data = json.load(file)

# Prepare data in question-answer format
qa_pairs = [
    {"question": entry["question"], "answer": entry["answer"]}
    for entry in data
    if "question" in entry and "answer" in entry
]

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Preprocess the data
def preprocess_data(qa_pairs):
    inputs, start_positions, end_positions = [], [], []

    for qa in qa_pairs:
        question = qa["question"]
        answer = qa["answer"]

        encoding = tokenizer(
            question,
            answer,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Find the start and end token positions of the answer in the input
        start_idx = input_ids.tolist().index(tokenizer.cls_token_id) + 1
        end_idx = len(input_ids) - input_ids.tolist()[::-1].index(tokenizer.sep_token_id) - 2

        inputs.append({"input_ids": input_ids, "attention_mask": attention_mask})
        start_positions.append(start_idx)
        end_positions.append(end_idx)

    return inputs, start_positions, end_positions

inputs, start_positions, end_positions = preprocess_data(qa_pairs)

# Convert data into tensors
input_ids = torch.stack([inp["input_ids"] for inp in inputs])
attention_masks = torch.stack([inp["attention_mask"] for inp in inputs])
start_positions = torch.tensor(start_positions)
end_positions = torch.tensor(end_positions)

# Train-test split
X_train, X_test, y_train_start, y_test_start, y_train_end, y_test_end = train_test_split(
    input_ids, start_positions, end_positions, test_size=0.2, random_state=42
)

attention_masks_train, attention_masks_test = train_test_split(attention_masks, test_size=0.2, random_state=42)

# Create DataLoaders
train_dataset = TensorDataset(X_train, attention_masks_train, y_train_start, y_train_end)
test_dataset = TensorDataset(X_test, attention_masks_test, y_test_start, y_test_end)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids_batch, attention_masks_batch, start_positions_batch, end_positions_batch = [
            item.to(device) for item in batch
        ]

        optimizer.zero_grad()

        # Forward pass with labels
        outputs = model(
            input_ids=input_ids_batch,
            attention_mask=attention_masks_batch,
            start_positions=start_positions_batch,
            end_positions=end_positions_batch,
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Save the model and tokenizer
model.save_pretrained("chatbot_model")
tokenizer.save_pretrained("chatbot_model")
print("Model and tokenizer saved.")

def predict_answer(question, context):
    # Tokenize the input question and context
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start = outputs.start_logits.argmax(dim=1)
        answer_end = outputs.end_logits.argmax(dim=1)

    # Decode the answer from token ids
    answer_tokens = inputs['input_ids'][0][answer_start:answer_end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    return answer

# Example: Predict the answer for a new query
context = "Here you can describe the context, such as project details or any other relevant data."
question = "What is the status of the project?"

answer = predict_answer(question, context)
print(f"Answer: {answer}")