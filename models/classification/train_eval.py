import argparse
import collections
import glob
import json
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import tqdm
import transformers

from contextlib import nullcontext
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score

import classification_lib

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    help="train eval or predict",
)
parser.add_argument(
    "-c",
    "--category",
    type=str,
    help="train eval or predict",
)

DEVICE = "cuda"
EPOCHS = 100
PATIENCE = 5
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
TRAIN, EVAL, PREDICT = "train eval predict".split()
PRE_TRAINED_MODEL_NAME = "bert-base-uncased"


random_seed_value = 131
random.seed(random_seed_value)
np.random.seed(random_seed_value)
torch.manual_seed(random_seed_value)
torch.cuda.manual_seed_all(random_seed_value)           # type: ignore
torch.backends.cudnn.deterministic = True   # type: ignore
torch.backends.cudnn.benchmark = False      # type: ignore

with open('label_map.json', 'r') as f:
  LABEL_MAP = json.load(f)

HistoryItem = collections.namedtuple(
    "HistoryItem", "epoch train_acc train_loss train_f1 val_acc val_loss val_f1".split())

Example = collections.namedtuple("Example", "identifier text target".split())


def get_label(original_label):
  return 0 if original_label == "none" else 1


tokenizer_fn = lambda tok, text: tok.encode_plus(
    text,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding="max_length",
    max_length=512,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)

EYE_2 = np.eye(2, dtype=np.float64)


class ClassificationDataset(Dataset):

  def __init__(self, filename, tokenizer, category, max_len=512):
    (
        self.texts,
        self.identifiers,
        self.target_indices,
    ) = classification_lib.get_examples(filename)

    num_classes = len(LABEL_MAP[category])
    EYE = np.eye(num_classes, dtype=np.float64)
    self.targets = [EYE[int(i)] for i in self.target_indices]
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, item):
    text = str(self.texts[item])
    target = self.targets[item]

    encoding = tokenizer_fn(self.tokenizer, text)

    return {
        "reviews_text": text,
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(),
        "targets": torch.tensor(target, dtype=torch.float64),
        "target_indices": self.target_indices[item],
        "identifier": self.identifiers[item],
    }


class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    output = self.drop(bert_output["pooler_output"])
    return self.out(output)


def create_data_loader(filename, tokenizer, category):
  ds = ClassificationDataset(
      filename,
      tokenizer,
      category,
  )
  return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4)


def build_data_loaders(data_dir, tokenizer, category):
  return (
      create_data_loader(
          f'{data_dir}/train.jsonl',
          tokenizer,
          category,
      ),
      create_data_loader(
          f'{data_dir}/dev.jsonl',
          tokenizer,
          category,
      ),
  )

def build_test_data_loader(data_dir, tokenizer, category):
  return create_data_loader(

          f'{data_dir}/test.jsonl',
          tokenizer,
          category,
  )

def train_or_eval(
    mode,
    model,
    data_loader,
    loss_fn,
    device,
    return_preds=False,
    optimizer=None,
    scheduler=None,
):
  assert mode in [TRAIN, EVAL]
  is_train = mode == TRAIN
  if is_train:
    model = model.train()
    context = nullcontext()
    assert optimizer is not None
    assert scheduler is not None
  else:
    model = model.eval()
    context = torch.no_grad()

  results_and_true_labels = []
  losses = []

  with context:
    for d in tqdm.tqdm(data_loader):
      input_ids, attention_mask, targets, target_indices = [
          d[k].to(device)
          for k in "input_ids attention_mask targets target_indices".split()
      ]

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
      results_and_true_labels += list(zip(*[preds.cpu().numpy().tolist(),
                              target_indices.cpu().numpy().tolist()]))
      loss = loss_fn(outputs, targets)
      losses.append(loss.item())
      if is_train:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

  if return_preds:
    return results_and_true_labels
  else:
    assert len(results_and_true_labels) == len(data_loader.dataset)
    acc, f1 = get_accuracies(results_and_true_labels)
    return acc, f1,  np.mean(losses)

def get_accuracies(results_and_true_labels):
  y_pred, y_true = zip(*results_and_true_labels)
  f1 = f1_score(y_true, y_pred, average="macro")
  accuracy = f1_score(y_true, y_pred, average="micro")
  return accuracy, f1

def do_train(tokenizer, model, loss_fn, data_dir, category):
  hyperparams = {
      "epochs": EPOCHS,
      "patience": PATIENCE,
      "learning_rate": LEARNING_RATE,
      "batch_size": BATCH_SIZE,
      "bert_model": PRE_TRAINED_MODEL_NAME,
      "category": category,
  }

  (
      train_data_loader,
      val_data_loader,
  ) = build_data_loaders(data_dir, tokenizer, category)

  ckpt_file = f'ckpt/{category}_best_model.bin'
  history_file = f'ckpt/{category}_history.pkl'

  optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = transformers.get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  history = []
  best_accuracy = 0
  best_accuracy_epoch = None

  for epoch in range(EPOCHS):

    if best_accuracy_epoch is not None and epoch - best_accuracy_epoch > PATIENCE:
      break

    print(f"Epoch {epoch + 1}/{EPOCHS} for {category}")
    if best_accuracy_epoch is not None:
      print(f"{epoch - best_accuracy_epoch} epochs since best accuracy")
    print("-" * 10)

    train_acc, train_f1, train_loss = train_or_eval(
        TRAIN,
        model,
        train_data_loader,
        loss_fn,
        DEVICE,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    val_acc, val_f1, val_loss = train_or_eval(EVAL, model, val_data_loader, loss_fn,
                                      DEVICE)

    history.append(HistoryItem(epoch, train_acc, train_loss, train_f1, val_acc,
    val_loss, val_f1))
    for k, v in history[-1]._asdict().items():
      print(k + "\t", v)
    print()

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), ckpt_file)
      best_accuracy = val_acc
      best_accuracy_epoch = epoch

  with open(history_file, "wb") as f:
    pickle.dump(history, f)


def do_eval(tokenizer, model, loss_fn, data_dir, category):
  test_data_loader = build_test_data_loader(
      data_dir,
      tokenizer,
      category
  )

  model.load_state_dict(torch.load(f"ckpt/{category}_best_model.bin"))

  acc, f1, loss = train_or_eval(EVAL, model, test_data_loader, loss_fn,
                                    DEVICE)

  print(f"{category} Test F1", f1)



def get_num_classes(data_dir, category):
  with open(f'{data_dir}/label_map.json', 'r') as f:
    return len(json.load(f)[category])

def main():

  args = parser.parse_args()
  assert args.mode in [TRAIN, EVAL, PREDICT]

  tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  num_classes = len(LABEL_MAP[args.category])
  model = SentimentClassifier(num_classes).to(DEVICE)
  loss_fn = nn.CrossEntropyLoss().to(DEVICE)

  data_dir = f'{args.data_dir}/{args.category}/'

  if args.mode == TRAIN:
    do_train(tokenizer, model, loss_fn, data_dir, args.category)
  elif args.mode == EVAL:
    do_eval(tokenizer, model, loss_fn, data_dir, args.category)
  elif args.mode == PREDICT:
    assert False


if __name__ == "__main__":
  main()
