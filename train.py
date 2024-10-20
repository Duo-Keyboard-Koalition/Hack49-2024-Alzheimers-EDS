import librosa
import torch
import torch.nn as nn
from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate
import numpy as np


# Step 1: Load and preprocess the dataset
def load_audio(file_path, sample_rate=16000):
    # Load audio file and resample to 16000 Hz
    speech_array, _ = librosa.load(file_path, sr=sample_rate)
    return speech_array


data = {
    'audio': [
        'Datasets/william1.wav',
        'Datasets/william.wav',
        'Datasets/amrnothealthy1.wav',
        'Datasets/amrhealthy.wav',
        'Datasets/williamHealthy3.wav',
        'Datasets/williamHealthy4.wav',
        'Datasets/williamHealthy5.wav',
        'Datasets/williamHealthy6.wav',
        'Datasets/williamHealthy7.wav',
        'Datasets/williamHealthy8.wav',
        'Datasets/williamHealthy9.wav',
        'Datasets/williamHealthy10.wav',
        'Datasets/williamHealthy11.wav',
        'Datasets/williamHealthy12.wav',
        'Datasets/williamHealthy13.wav',
        'Datasets/williamHealthy14.wav',
        'Datasets/williamHealthy15.wav',
        'Datasets/williamHealthy16.wav',
        'Datasets/williamHealthy17.wav',
        'Datasets/williamHealthy18.wav',
        'Datasets/williamHealthy19.wav',
        'Datasets/william_healthy2.wav',
        'Datasets/williamNotHealthy1.wav',
        'Datasets/williamNotHealthy2.wav',
        'Datasets/williamNotHealthy3.wav',
        'Datasets/williamNotHealthy4.wav',
        'Datasets/williamNotHealthy5.wav',
        'Datasets/williamNotHealthy6.wav',
    ],
    'label': [1, 0, 1, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]  # Corresponding labels
}

# Create Dataset from the dictionary
dataset = Dataset.from_dict(data)

# Load Wav2Vec2 processor for feature extraction
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

# Step 2: Preprocessing function to apply the processor to the dataset
def preprocess_function(examples):
    audio = [load_audio(path) for path in examples['audio']]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess_function, batched=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 3: Modify Wav2Vec2 model for binary classification
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=2,  # Binary classification
    problem_type="single_label_classification"
)

model.to(device)

# Step 4: Define metrics and evaluation function
metric = evaluate.load("accuracy")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = metric.compute(predictions=preds, references=labels)
    return accuracy

# Step 5: Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    warmup_steps=0,
    logging_steps=10,
    gradient_accumulation_steps=4,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    learning_rate=0.1
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # Assuming you split the data later into train/val sets
    tokenizer=processor.feature_extractor,  # Feature extractor (processor)
    compute_metrics=compute_metrics
)

# Step 7: Train the model
trainer.train()

# Step 8: Inference on new audio files
def predict(model, audio_path):
    # Load and preprocess the new audio file
    audio = load_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Output class
    return "Yes" if predicted_class == 1 else "No"

# Example usage of inference
new_audio_path = 'Datasets/william_healthy.wav'
prediction = predict(model, new_audio_path)
print(f"Prediction: Alzheimer's = {prediction}")