from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader, Dataset
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math
import json
from sklearn.model_selection import train_test_split

with open('scripts/trainCrossEncoder/train_samples.json', 'r') as f:
    train_samples = json.load(f)

train_samples, dev_samples = train_test_split(train_samples, test_size=0.1, random_state=42)
train_input_examples = [InputExample(texts=sample["texts"], label=sample["label"]) for sample in train_samples]
dev_input_examples = [InputExample(texts=sample["texts"], label=sample["label"]) for sample in dev_samples]

model = CrossEncoder('distilroberta-base', num_labels=1)

train_dataloader = DataLoader(train_input_examples, shuffle=True, batch_size=16) # type: ignore

num_epochs = 5
# We add an evaluator, which evaluates the performance during training
evaluator = CECorrelationEvaluator.from_input_examples(dev_input_examples, name='dev')
# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
model_save_path = 'output/crossencoder'

model.fit(
    train_dataloader=train_dataloader,
    evaluator= evaluator, # type: ignore
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=model_save_path
)