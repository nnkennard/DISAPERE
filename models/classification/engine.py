import torch
import torch.nn as nn
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.ERROR)


def loss_fn(outputs, targets):
  return nn.CrossEntropyLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer, device, scheduler, epoch):
  model.train()
  tr_loss = 0
  nb_tr_steps = 0
  for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
    ids = d["ids"]
    token_type_ids = d["token_type_ids"]
    mask = d["mask"]
    targets = d["targets"]

    ids = ids.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    targets = targets.to(device, dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    loss = loss_fn(outputs, targets)
    tr_loss += loss.item()
    nb_tr_steps += 1

    loss.backward()
    optimizer.step()
    scheduler.step()
  epoch_loss = tr_loss / nb_tr_steps
  print(f"Training Loss for Epoch: {epoch} {epoch_loss}")


def eval_fn(data_loader, model, device, epoch):
  model.eval()
  tr_loss = 0
  nb_tr_steps = 0
  fin_targets = []
  fin_outputs = []
  with torch.no_grad():
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
      ids = d["ids"]
      token_type_ids = d["token_type_ids"]
      mask = d["mask"]
      targets = d["targets"]

      ids = ids.to(device, dtype=torch.long)
      token_type_ids = token_type_ids.to(device, dtype=torch.long)
      mask = mask.to(device, dtype=torch.long)
      targets = targets.to(device, dtype=torch.long)

      outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

      loss = loss_fn(outputs, targets)
      tr_loss += loss.item()
      nb_tr_steps += 1

      fin_targets.extend(targets.cpu().detach().numpy().tolist())
      fin_outputs.extend(
          torch.argmax(outputs, dim=1).cpu().detach().numpy().tolist())

    epoch_loss = tr_loss / nb_tr_steps
    print(f"Validation Loss Epoch: {epoch} {epoch_loss}")

  return fin_outputs, fin_targets
