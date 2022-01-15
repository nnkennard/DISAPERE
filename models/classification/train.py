import pickle
import csv
import collections
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import sys
import transformers

from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import classification_lib
from classification_lib import Config


def get_num_train_steps(sample_list):
  return int(len(sample_list) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)


def main():

  data_file, task = sys.argv[1:3]

  with open("data/label_map.pkl", 'rb') as f:
    label_map = pickle.load(f)

  label2int = label_map["coarse"]

  model_path = "models/" + task + "_best.pt"
  samples = collections.defaultdict(list)
  dataframe_builders = collections.defaultdict(list)
  with open(data_file, 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t')
    for row in reader:
      if not row['dataset'] == task:
        continue
      label_id = label2int[row['label']]
      samples[row['split']].append((row['sentence1'], label_id))
      dataframe_builders[row['split']].append({
          'sentence': row['sentence1'],
          'task': task,
          'label': row['label'],
          'encoded': float(label_id)
      })

  dataframes = {
      split: pd.DataFrame.from_dict(dicts)
      for split, dicts in dataframe_builders.items()
  }

  pd.set_option('display.max_columns', None)

  dataloaders = {}
  for subset, sample_list in samples.items():
    if subset == 'train':
      batch_size = Config.TRAIN_BATCH_SIZE
      num_workers = 4
    else:
      batch_size = Config.VALID_BATCH_SIZE
      num_worker = 1

    dataset = classification_lib.BERTDataset(sample_list)
    dataloaders[subset] = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      num_workers=num_workers)

  device = torch.device(Config.DEVICE)
  model = classification_lib.BERTBaseUncased(len(label2int))
  model.to(device)

  param_optimizer = list(model.named_parameters())
  no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
  optimizer_parameters = [
      {
          "params": [
              p for n, p in param_optimizer
              if not any(nd in n for nd in no_decay)
          ],
          "weight_decay": 0.001,
      },
      {
          "params": [
              p for n, p in param_optimizer if any(nd in n for nd in no_decay)
          ],
          "weight_decay": 0.0,
      },
  ]

  optimizer = AdamW(optimizer_parameters, lr=3e-5)
  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=get_num_train_steps(samples["train"]))

  best_val_accuracy = float('-inf')
  best_val_epoch = None

  best_accuracy = 0
  for epoch in range(Config.EPOCHS):
    engine.train_fn(dataloaders["train"], model, optimizer, device, scheduler,
                    epoch)
    outputs, targets = engine.eval_fn(dataloaders['dev'], model, device, epoch)
    accuracy = metrics.accuracy_score(outputs, targets)
    print(f"Validation Accuracy  = {accuracy}")
    if accuracy > best_val_accuracy:
      torch.save(model.state_dict(), model_path)
      best_val_accuracy = accuracy
      best_val_epoch = epoch
      print("Best val accuracy till now {}".format(best_val_accuracy))

    if best_val_epoch < (epoch - Config.PATIENCE):
      break

  model.load_state_dict(torch.load(model_path))
  for subset in ['train', 'dev', 'test']:
    outputs, targets = engine.eval_fn(dataloaders[subset], model, device, epoch)

    result_df_dicts = []
    for o, t in zip(outputs, targets):
      result_df_dicts.append({"output": o, "target": t})

    result_df = pd.DataFrame.from_dict(result_df_dicts)

    final_df = pd.concat([dataframes[subset], result_df], axis=1)
    #assert all(i.encoded == i.target for i in final_df.itertuples())

    result_file = "results/" + subset + "_" + task + ".csv"
    final_df.to_csv(result_file)


if __name__ == "__main__":
  main()
