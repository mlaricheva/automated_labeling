import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler
import scipy
import random
from collections import defaultdict
from tqdm import tqdm, trange
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification, AutoConfig, TrainingArguments, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os

global device, device_ids
n_gpu = torch.cuda.device_count()
device_ids = GPUtil.getAvailable(limit = 4)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, lab2ind):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = self.data.content
        self.labels = self.data.label
        self.max_len = max_len
        self.lab2ind = lab2ind
        self.all_indx = list(set(lab2ind.values()))
        self.label_binarizer = MultiLabelBinarizer(self.all_indx)
        self.pad_ind = tokenizer.pad_token_id  # padding

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        content = str(self.comment_text[index])

        tokenized_texts = tokenizer.tokenize(content)
        tokenized_texts = tokenized_texts[-min((self.max_len), len(tokenized_texts)):]
        input_ids = tokenizer.encode(tokenized_texts, add_special_tokens=True)

        input_ids = pad_sequences([input_ids], maxlen=self.max_len + 2, dtype="long", truncating="pre", padding="pre",
                                  value=self.pad_ind)[0]

        attention_mask = [
            1 if i != tokenizer.convert_tokens_to_ids("<pad>") and i != tokenizer.convert_tokens_to_ids("PAD") else 0
            for i in input_ids]

        label = [label2ind[x] for x in self.labels[index].split(",")]
        label = self.label_binarizer.fit_transform([labels])[0]
        return {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(attention_mask, dtype=torch.float32),
            'targets': torch.tensor(label, dtype=torch.long)
        }


# define a function for data preparation
def regular_encode(train_file_path, test_file_path, tokenizer, lab2ind, num_workers=2, batch_size=64, maxlen=32,
                   validation_split=.1, random_seed=42):
    dataset_params = {'batch_size': batch_size, 'num_workers': num_workers}

    ## TRAIN
    df = pd.read_csv(train_file_path, header=0, names=['content', 'label'])
    # Split the dataframe
    np.random.seed(random_seed)
    train, evaluation = train_test_split(df, test_size=validation_split, shuffle=True, random_state=random_seed)
    train = train.reset_index(drop=True)
    evaluation = evaluation.reset_index(drop=True)

    # Creating custom dataset objects
    custom_train = CustomDataset(train, tokenizer, maxlen, lab2ind)
    custom_eval = CustomDataset(evaluation, tokenizer, maxlen, lab2ind)
    train_loader = DataLoader(custom_train, **dataset_params)
    validation_loader = DataLoader(custom_eval, **dataset_params)

    print("{} Train size: {}".format("val", len(train_loader.dataset)))
    print("{} Validation size: {}".format("val", len(validation_loader.dataset)))

    ## TEST
    df_test = pd.read_csv(test_file_path, header=0, names=['content', 'label'])

    # Creating custom dataset object
    test_dataset = CustomDataset(df_test, tokenizer, maxlen, lab2ind)
    test_loader = DataLoader(test_dataset, **dataset_params)
    print(df.shape)
    print("{} Dataset: {}".format("train", len(train_loader.dataset)))
    print("{} Dataset: {}".format("test", len(test_loader.dataset)))
    print("{} Test size: {}".format("val", len(test_loader.dataset)))

    return train_loader, validation_loader, test_loader


def encode_test(train_file_path, test_file_path, tokenizer, lab2ind, num_workers=2, batch_size=64, maxlen=32,
                validation_split=.1, random_seed=42):
    dataset_params = {'batch_size': batch_size, 'num_workers': num_workers}

    ## TRAIN
    df = pd.read_csv(train_file_path, header=0, names=['content', 'label'])
    df_test = pd.read_csv(test_file_path, header=0, names=['content', 'label'])

    # Creating custom dataset objects
    custom_train = CustomDataset(df, tokenizer, maxlen, lab2ind)
    test_dataset = CustomDataset(df_test, tokenizer, maxlen, lab2ind)
    train_loader = DataLoader(custom_train, **dataset_params)
    test_loader = DataLoader(test_dataset, **dataset_params)

    print(df.shape)
    print("{} Dataset: {}".format("train", len(train_loader.dataset)))
    print("{} Dataset: {}".format("test", len(test_loader.dataset)))

    return train_loader, test_loader


class Bert_Multilabel_Cls(nn.Module):
    def __init__(self, lab2ind, bert_model, hidden_size):
        super(Bert_Multilabel_Cls, self).__init__()
        self.hidden_size = hidden_size
        self.bert_model = bert_model
        self.label_num = len(lab2ind)

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.hidden_size, self.label_num)

        # non-linear layer initial
        torch.nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, bert_mask):
        output = self.bert_model(input_ids=input_ids, attention_mask=bert_mask)
        pooler_output = output[1]

        x = self.dense(pooler_output)
        x = torch.tanh(x)
        x = self.dropout(x)

        fc_output = self.fc(x)
        fc_output = F.sigmoid(fc_output)
        return fc_output

def train(model, iterator, optimizer, scheduler, device):
    model.train()
    epoch_loss = 0.0

    for _, batch in enumerate(tqdm(iterator, desc="Train Iteration")):

        input_ids = batch['ids'].to(device, dtype=torch.long)  # [batch_size,128]
        input_mask = batch['mask'].to(device, dtype=torch.float32)  # [batch_size,128]
        labels = batch['targets'].to(device, dtype=torch.long)  # [batch_size,1]

        ## check the labels size
        outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=labels)
        loss = criterion(logits, labels)

        # delete used variables to free GPU memory
        del batch, input_ids, input_mask, labels
        optimizer.zero_grad()

        if torch.cuda.device_count() == 1 or device == 'cpu':
            loss.backward()
            epoch_loss += loss.cpu().item()
        else:
            loss.mean().backward()
            epoch_loss += loss.mean().cpu().item()

        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.step()

    # free GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, device):
    model.eval()
    epoch_loss = 0.0
    all_pred = []
    all_label = []
    with torch.no_grad():

        for _, batch in enumerate(tqdm(iterator, desc="Eval Iteration")):

            input_ids = batch['ids'].to(device, dtype=torch.long)
            input_mask = batch['mask'].to(device, dtype=torch.float32)
            labels = batch['targets'].to(device, dtype=torch.long)

            outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=labels)
            logits = outputs.logits
            loss = criterion(logits, labels)

            # delete used variables to free GPU memory
            del batch, input_ids, input_mask

            if torch.cuda.device_count() == 1 or device == 'cpu':
                epoch_loss += loss.cpu().item()
            else:
                epoch_loss += loss.mean().cpu().item()

            # identify the predicted class for each example in the batch
            output = logits.cpu()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0  ## assign 0 label to those with less than 0.5

            # put all the true labels and predictions to two lists
            all_pred.extend(predicted.numpy())
            all_label.extend(labels.cpu().detach().numpy())

    accuracy = accuracy_score(all_label, all_pred)
    f1score_mac = f1_score(all_label, all_pred, average='macro')
    f1score_weight = f1_score(all_label, all_pred, average='weighted')
    return epoch_loss / len(iterator), accuracy, f1score_mac, f1score_weight


def create_optimizer_and_scheduler(model, num_training_steps, warmup_steps, weight_decay, learning_rate,
                                   is_constant_lr=False):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate
    )

    if is_constant_lr == True:
        lr_scheduler = get_constant_schedule(optimizer)
    else:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
    return optimizer, lr_scheduler

def model_parallel(model, device_ids):
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            print("1 gpu")
            model = model.to(device)
        else:
            print("more gpus")
            torch.backends.cudnn.benchmark = True
            model = model.to(device)
            model = nn.DataParallel(model, device_ids=device_ids)
    else:
        model = model
    return(model)

def main():
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    ## Set seed of randomization and working device
    manual_seed = 77
    torch.manual_seed(manual_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained("mlaricheva/roberta-psych")

    ### TRAIN DATA PREPROCESSING (labels)
    with open('label2ind.json') as f:  # replace contractions
        label2ind = json.load(f)
    train_dataloader, val_dataloader, test_dataloader = regular_encode(
        TRAIN_PATH,
        TEST_PATH,
        tokenizer, label2ind, num_workers=1, maxlen=126, batch_size=4)

    bert_model = RobertaModel.from_pretrained("mlaricheva/roberta-psych", num_labels=2)
    model = Bert_Multilabel_Cls(label2ind, bert_model, 768)
    model = model_parallel(model, device_ids)

    config = {
        "lr": 1e-5,
        "batch_size": 8,
        "epochs": 30
    }

    epoch_test_res = []
    lr = config["lr"]
    batch_size = config["batch_size"]
    max_grad_norm = 1.0
    weight_decay = 0.01
    epochs = config["epochs"]
    warmup_proportion = 0.1

    train_dataloader, test_dataloader = encode_test(
        TRAIN_PATH,
        TEST_PATH,
        tokenizer, label2ind, num_workers=1, maxlen=126, batch_size=batch_size, random_seed=manual_seed)
    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = num_training_steps * warmup_proportion

    for i in range(3):
        manual_seed = random.randint(1, 1000)
        torch.manual_seed(manual_seed)
        bert_model = RobertaModel.from_pretrained("mlaricheva/roberta-psych", num_labels=2)
        model = Bert_Multilabel_Cls(label2ind, bert_model, 768)
        model = model_parallel(model, device_ids)
        optimizer, scheduler = create_optimizer_and_scheduler(model, num_training_steps, num_warmup_steps, weight_decay, lr)

        for epoch in trange(epochs, desc="Epoch"):
            train_loss = train(model, train_dataloader, optimizer, scheduler, device)
            test_loss, accuracy, f1score_mac, f1score_weight = evaluate(model, test_dataloader, device)
            res = {"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss, "accuracy": accuracy,
                   "f1score_mac": f1score_mac, "f1score_weight": f1score_weight}
            if epoch + 1 == 19:
                with open(RESULTS_FILE, 'a+') as batch_res_file:
                    batch_res_file.write(res)
                break
        if device == 'cuda':
            torch.cuda.empty_cache()
        epoch_test_res.append(res)

if __name__ == "__main__":
    main()