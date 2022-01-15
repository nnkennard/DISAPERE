import transformers
import torch
import torch.nn as nn


class Config(object):
  DEVICE = "cpu"
  MAX_LEN = 64
  TRAIN_BATCH_SIZE = 32
  VALID_BATCH_SIZE = 32
  EPOCHS = 100
  BERT_PATH = "bert-base-uncased"
  MODEL_PATH = "./classification.bin"
  TRAINING_FILE = "./review-sentence_train_head.csv"
  TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH,
                                                         do_lower_case=True)
  PATIENCE = 5


class BERTBaseUncased(nn.Module):

  def __init__(self, num_classes):
    super(BERTBaseUncased, self).__init__()
    self.bert = transformers.BertModel.from_pretrained(Config.BERT_PATH)
    self.pre_classifier = nn.Linear(768, 768)
    self.dropout = nn.Dropout(0.3)
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, ids, mask, token_type_ids):
    output_1 = self.bert(ids,
                         attention_mask=mask,
                         token_type_ids=token_type_ids)
    hidden_state = output_1[0]
    pooler = hidden_state[:, 0]
    pooler = self.pre_classifier(pooler)
    pooler = nn.ReLU()(pooler)
    pooler = self.dropout(pooler)
    output = self.classifier(pooler)
    return output


class BERTDataset:

  def __init__(self, sample_list):
    reviews, targets = zip(*sample_list)
    self.review = reviews
    self.target = targets
    self.tokenizer = Config.TOKENIZER
    self.max_len = Config.MAX_LEN

  def __len__(self):
    return len(self.review)

  def __getitem__(self, item):
    review = str(self.review[item])
    review = " ".join(review.split())

    inputs = self.tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=self.max_len,
        pad_to_max_length=True,
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "targets": torch.tensor(self.target[item], dtype=torch.float),
    }
