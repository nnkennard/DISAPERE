from torch.utils.data import DataLoader
import collections
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import os
import torch
import numpy as np
import random
from tqdm import tqdm
import json

random_seed_value = 131
random.seed(random_seed_value)
np.random.seed(random_seed_value)
torch.manual_seed(random_seed_value)
torch.cuda.manual_seed_all(random_seed_value)           # type: ignore
torch.backends.cudnn.deterministic = True   # type: ignore
torch.backends.cudnn.benchmark = False      # type: ignore



distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

# Negative pairs should have a distance of at least 0.5
margin = 5


def build_dataset(data_dir, subset):
  input_examples = []
  with open(f'{data_dir}/{subset}.jsonl', 'r') as f:
    for line in f:
      example = json.loads(line)
      for reb_i, reb_sent in enumerate(example["rebuttal_sentences"]):
        for rev_i, rev_sent in enumerate(example["review_sentences"]):
          input_examples.append(InputExample(
            texts=[reb_sent, rev_sent],
            label=example['alignment'][reb_i][rev_i]
            ))
  return input_examples

def build_data_loader(dataset):
  return DataLoader(dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE)

def build_binary_evaluator(data_loader):
  texts_1 = []
  texts_2 = []
  labels = []
  for example in data_loader.dataset:
    text_1, text_2 = example.texts
    texts_1.append(text_1)
    texts_2.append(text_2)
    labels.append(example.label)

  return evaluation.BinaryClassificationEvaluator(texts_1, texts_2, labels)

NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 128

# Train the model
def main():
  data_dir = "prepared_data/"

  datasets = {
  subset: build_dataset(data_dir, subset)
  for subset in "train dev test".split() }


  dev_data_loader = build_data_loader(datasets['dev'])


  model = SentenceTransformer('all-MiniLM-L6-v2')
  model_save_path = f'outputs/best_model.bin'
  os.makedirs(model_save_path, exist_ok=True)

  evaluators = [build_binary_evaluator(dev_data_loader)]
  seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
  train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

  model.fit(train_objectives=[(build_data_loader(datasets['train']), train_loss)],
            evaluator=seq_evaluator,
            epochs=NUM_EPOCHS,
            warmup_steps=1000,
            output_path=model_save_path
            )


if __name__ == "__main__":
  main()
