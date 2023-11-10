from transformers import BertModel, AutoConfig

import torchmetrics
import matplotlib.backends
matplotlib.use('TkAgg')

import torch
import torch.nn as nn

import wandb
import matplotlib.pyplot as plt
from base_model.ChEMUDataModule import *


class NERModelWithCRF(nn.Module):
    def __init__(self, num_labels):
        super(NERModelWithCRF, self).__init__()
        configuration = AutoConfig.from_pretrained('bert-base-uncased')
        configuration.hidden_dropout_prob = 0.5
        configuration.attention_probs_dropout_prob = 0.1

        # Input Layer (BertModel)
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=configuration)

        # Linear Layer
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)

        # CRF Layer
        # self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask):
        # BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # Linear Layer
        logits = self.linear(outputs.last_hidden_state)

        # print(logits.shape) # Emission score tensor of size (batch_size, seq_length, num_tags)
        # print(attention_mask.shape) # Mask tensor of size (batch_size, seq_length)

        # CRF Layer
        # tags = self.crf.decode(logits, mask=attention_mask)

        return logits  # , tags


class EncoderNERModel(pl.LightningModule):
    def __init__(self, accumulation_schedule):
        super().__init__()
        self.max_classes = 10
        self.model = NERModelWithCRF(self.max_classes)  # .to(torch.bfloat16)

        self.batch_size = 3
        self.accumulation_schedule = accumulation_schedule

        self.logit_scaling = nn.Parameter(torch.tensor([
            9.7547e-03, 1.8360e-01, 5.1081e-01, 8.5402e-03, 1.1254e-01, 5.5558e-03,
            1.8673e-02, 6.2147e-03, 1.0438e-02, 7.4143e-03, 1.1269e-02, 5.2018e-03,
            5.6984e-03, 1.4942e-02, 2.1928e-02, 3.2298e-02, 5.2117e-03, 1.1185e-02,
            4.6905e-03, 6.5982e-03, 4.3463e-03, 2.2125e-03, 5.4083e-05, 7.3258e-04,
            9.3417e-05]), requires_grad=False)

        # set weight of everything to 1 and class 2 to 0.0125
        self.weights = torch.ones(self.max_classes)
        self.weights[0] = 0.0125

        self.criterion = nn.CrossEntropyLoss(weight=self.weights, reduction='mean', ignore_index=-100,
                                             label_smoothing=0.1)

        self.f1 = torchmetrics.F1Score(num_classes=self.max_classes, task="multiclass", average=None, )
        self.f1_micro = torchmetrics.F1Score(num_classes=self.max_classes, task="multiclass", average='micro', )

        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.max_classes, task="multiclass", normalize='true')

        # self.save_hyperparameters() #save all hyperparameters to wandb

    def forward(self, input, attention_mask, labels=None, train=True):
        logits = self.model(input, attention_mask)
        loss = 0
        if labels is not None:
            loss = self.custom_loss_function(logits, labels, attention_mask, train)

            # #log f1 score to wandb
            # self.log("train_f1" if train else "valid_f1", self.f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)

            if not train:
                # log loss
                self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         batch_size=self.batch_size, sync_dist=True)

        return loss

    def custom_loss_function(self, logits, labels, attention_mask, train=True):
        # apply logit adjustment
        # logits = logits + torch.log(self.logit_scaling**1 + 1e-12)

        # loss = -1 * self.model.crf(logits, labels, mask=attention_mask, reduction='mean')
        b = 0.1

        loss = self.criterion(logits.view(-1, self.max_classes), labels.view(-1))

        if train:
            loss = (loss - b).abs() + b  # b is the flooding level.

        preds = torch.argmax(logits, dim=-1)

        # Mask out the padding tokens
        # active_logits = logits.view(-1, logits.shape[-1])[attention_mask.view(-1) == 1]
        active_labels = labels.view(-1)[attention_mask.view(-1) == 1]
        active_preds = preds.view(-1)[attention_mask.view(-1) == 1]

        # sum up
        # loss = torch.sum(loss)

        # if not train:
        # calculate F1 score using torchmetrics
        self.f1.update(active_preds, active_labels)
        self.f1_micro.update(active_preds, active_labels)
        self.confusion_matrix.update(active_preds, active_labels)

        return loss

    def training_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch
        loss = self(input_ids, attn_mask, labels, train=True)
        return {"loss": loss, 'unscaled_lmao': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch
        loss = self(input_ids, attn_mask, labels, train=False)
        return {"valid_loss": loss}

    def test_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch
        loss = self(input_ids, attn_mask, labels, train=False)
        return {"test_loss": loss}

    def on_train_batch_end(self, out, batch, batch_idx) -> None:
        # DO NOT TOUCH
        self.log("train_loss", out['unscaled_lmao'], on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

    def on_train_epoch_end(self):
        # Compute per-class F1 at the end of epoch
        # f1_per_class = self.f1.compute()

        # # Log average F1 score
        # self.log('train_avg_f1', f1_per_class.mean(), on_epoch=True, prog_bar=True)

        # # Log each per-class F1 score
        # for i, f1_val in enumerate(f1_per_class):
        #     self.log(f'train_f1_class_{i}', f1_val, on_epoch=True)

        # # Compute and log confusion matrix at the end of epoch
        # conf_matrix = self.confusion_matrix.compute()
        # self.log_confusion_matrix(conf_matrix, 'train')

        # Reset metrics
        self.f1.reset()
        self.confusion_matrix.reset()

    def log_confusion_matrix(self, fig, ax, stage):
        # Convert confusion matrix to a plot
        # fig, ax = plt.subplots(figsize=(12, 12))
        # cax = ax.matshow(conf_matrix.cpu().numpy())
        # plt.title(f'Confusion matrix for {stage}')
        # fig.colorbar(cax)

        # Log to WandB
        wandb.log({f'{stage}_confusion_matrix': [wandb.Image(fig)]})

        # Close the figure to free up memory
        plt.close(fig)

    def on_validation_epoch_end(self):
        # Compute per-class F1 at the end of epoch
        f1_per_class = self.f1.compute()

        # Log average F1 score
        self.log('val_avg_f1', f1_per_class.mean(), on_epoch=True, prog_bar=True)
        self.log('valid_f1', self.f1_micro.compute(), on_epoch=True, prog_bar=True)

        # Log each per-class F1 score
        for i, f1_val in enumerate(f1_per_class):
            self.log(f'val_f1_class_{i}', f1_val, on_epoch=True)

        # Compute and log confusion matrix at the end of epoch
        fig, ax = self.confusion_matrix.plot(add_text=False)
        self.log_confusion_matrix(fig, ax, 'valid')

        self.f1.reset()
        self.f1_micro.reset()
        self.confusion_matrix.reset()

    def configure_optimizers(self):
        # get num epochs from trainer
        num_epochs = self.trainer.max_epochs

        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=1e-4, weight_decay=0.1, )  # fused=True)
        # optimizer = Lion(self.trainer.model.parameters(), lr=3e-4/3, weight_decay=0.1*15, use_triton=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                        total_steps=self.trainer.estimated_stepping_batches,
                                                        pct_start=0.3)

        # return optimizer
        return [optimizer], [{'scheduler': scheduler, "interval": "step"}]
