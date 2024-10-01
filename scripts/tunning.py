import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report


class Prepocess:

  def read_conll_file(self, file_path):
      with open(file_path, "r") as f:
          content = f.read().strip()
          sentences = content.split("\n\n")
          data = []
          for sentence in sentences:
              tokens = sentence.split("\n")
              token_data = []
              for token in tokens:
                  token_data.append(token.split())
              data.append(token_data)
      return data

  def convert_to_dataset(self, data, label_map, chunk_size=15):
      formatted_data = {"tokens": [], "ner_tags": []}

      for sentence in data:
          tokens = [token_data[0] for token_data in sentence]
          ner_tags = [label_map[token_data[1]] for token_data in sentence]

          # Split tokens and ner_tags into chunks
          for i in range(0, len(tokens), chunk_size):
              chunk_tokens = tokens[i:i + chunk_size]
              chunk_ner_tags = ner_tags[i:i + chunk_size]
              formatted_data["tokens"].append(chunk_tokens)
              formatted_data["ner_tags"].append(chunk_ner_tags)

      return Dataset.from_dict(formatted_data)


  def process(self, filepath):
      data = self.read_conll_file(filepath)

      label_list = sorted(list(set([token_data[1] for sentence in data for token_data in sentence])))
      label_map = {label: i for i, label in enumerate(label_list)}

      dataset = self.convert_to_dataset(data, label_map, chunk_size=15)

      datasets = dataset.train_test_split(test_size=0.2)

      train_valid = datasets['train'].train_test_split(test_size=0.2)
      final_datasets = DatasetDict({
                                  'train': train_valid['train'],
                                  'validation': train_valid['test'],
                                  'test': datasets['test']
                                  })

      return final_datasets
