from transformers import BertModel, AutoConfig

import torchmetrics
import matplotlib.backends
matplotlib.use('TkAgg')

import torch
import torch.nn as nn

import wandb
import matplotlib.pyplot as plt
from continual_data_module import *
import copy

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
        
        self.model = torch.compile(self.model)

        self.batch_size = 32
        self.accumulation_schedule = accumulation_schedule
            
        #continual crap
        self.continual_step = 0
        self.tensor_dir = "continual_step_tensors"
        self.setup_test()

        self.continual_classes_schedule = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10}
        self.continual_epochs_schedule = {50 + 20*i:i for i in range(len(self.continual_classes_schedule))}
        print(self.continual_epochs_schedule)


        current_classes = self.continual_classes_schedule[self.continual_step]

        # set weight of everything to 1 and class 2 to 0.0125
        self.weights = torch.ones(current_classes)
        self.weights[0] = 0.125

        self.criterion = nn.CrossEntropyLoss(weight=self.weights, reduction='mean', ignore_index=-100,
                                            label_smoothing=0.1)

        self.f1 = torchmetrics.F1Score(num_classes=current_classes, task="multiclass", average=None, )
        self.f1_micro = torchmetrics.F1Score(num_classes=current_classes, task="multiclass", average='micro', )
        self.teacher = None
        
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=current_classes, task="multiclass", normalize='true')
        # self.save_hyperparameters() #save all hyperparameters to wandb

    def forward(self, input, attention_mask, labels=None, train=True):
        logits = self.model(input, attention_mask)
        teacher_logits = None
        if train and self.teacher is not None:
            with torch.no_grad():
                teacher_logits = self.teacher(input, attention_mask)
        loss = 0
        if labels is not None:
            loss = self.custom_loss_function(logits, labels, attention_mask, train, teacher_logits=teacher_logits)

            # #log f1 score to wandb
            # self.log("train_f1" if train else "valid_f1", self.f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)

            if not train:
                # log loss
                self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         batch_size=self.batch_size, sync_dist=True)

        return loss

    # def custom_loss_function(self, logits, labels, attention_mask, train=True, teacher_logits=None):
    #     # apply logit adjustment
    #     # logits = logits + torch.log(self.logit_scaling**1 + 1e-12)

    #     # loss = -1 * self.model.crf(logits, labels, mask=attention_mask, reduction='mean')
    #     b = 0.1

    #     loss = self.criterion(logits.view(-1, self.continual_classes_schedule[self.continual_step]), labels.view(-1))

    #     # if train:
    #     #     loss = (loss - b).abs() + b  # b is the flooding level.

    #     preds = torch.argmax(logits, dim=-1)

    #     # Mask out the padding tokens
    #     # active_logits = logits.view(-1, logits.shape[-1])[attention_mask.view(-1) == 1]
    #     active_labels = labels.view(-1)[attention_mask.view(-1) == 1]
    #     active_preds = preds.view(-1)[attention_mask.view(-1) == 1]

    #     # sum up
    #     # loss = torch.sum(loss)

    #     # if not train:
    #     # calculate F1 score using torchmetrics
    #     self.f1.update(active_preds, active_labels)
    #     self.f1_micro.update(active_preds, active_labels)
    #     self.confusion_matrix.update(active_preds, active_labels)

    #     return loss

    def custom_loss_function(self, logits, labels, attention_mask, train, teacher_logits,):
        current_classes = self.continual_classes_schedule[self.continual_step]

        # Select logits for current classes
        logits = logits[:, :, :current_classes]

        if teacher_logits is not None and train:
            labels_mask = labels.clone()
            ignore_mask = (labels == -100)
            labels_mask[ignore_mask] = 0

            previous_classes = current_classes - 1
            teacher_logits = teacher_logits[:, :, :previous_classes]
            teacher_probs = torch.nn.functional.softmax(teacher_logits*5, dim=-1)

            one_hot_labels = torch.nn.functional.one_hot(labels_mask, num_classes=current_classes).float()
            zero_label_mask = (labels_mask == 0)# & ~ignore_mask
            
            temp = one_hot_labels[zero_label_mask]
            temp[:, :previous_classes] = teacher_probs[zero_label_mask]
            one_hot_labels[zero_label_mask] = temp #hack around pytorch syntax

            one_hot_labels[ignore_mask] = 0

            loss = self.criterion(logits.transpose(1, 2), one_hot_labels.transpose(1, 2))

            # Apply mask to loss and compute mean
            loss = loss * (~ignore_mask).float()
            loss = loss.mean()

        else:
            # Compute loss directly with integer labels
            loss = self.criterion(logits.view(-1, self.continual_classes_schedule[self.continual_step]), labels.view(-1))

            

        # Update metrics if not training
        if not train:
            preds = torch.argmax(logits, dim=-1)
            active_labels = labels.view(-1)[attention_mask.view(-1) == 1]
            active_preds = preds.view(-1)[attention_mask.view(-1) == 1]

            #filter out -100 from both activate labels and preds
            active_preds = active_preds[active_labels != -100]
            active_labels = active_labels[active_labels != -100]
            self.f1.update(active_preds, active_labels)
            self.f1_micro.update(active_preds, active_labels)
            self.confusion_matrix.update(active_preds, active_labels)
        
        #check loss dimensions and reduce as needed
        if len(loss.shape) > 0:
            loss = loss.mean()
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
    
    def setup_test(self):
        # Assign train and validation datasets for use in dataloaders
        self.chemu_train = torch.load(os.path.join(self.tensor_dir, str(self.continual_step), 'train_tensor.pth'))
        self.chemu_val = torch.load(os.path.join(self.tensor_dir, str(self.continual_step), 'val_tensor.pth'))
        self.chemu_test = torch.load(os.path.join(self.tensor_dir, str(self.continual_step), 'test_tensor.pth'))

    def train_dataloader(self):
        return DataLoader(self.chemu_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=8, prefetch_factor=2, drop_last=True,)
                        #   persistent_workers=True, pin_memory=True,)

    def val_dataloader(self):
        return DataLoader(self.chemu_test, batch_size=self.batch_size,
                          num_workers=8, prefetch_factor=2, drop_last=False,)
                        #   persistent_workers=True, pin_memory=True,)

    def test_dataloader(self):
        return DataLoader(self.chemu_test, batch_size=self.batch_size,
                          num_workers=8, prefetch_factor=2, drop_last=False,)
                        #   persistent_workers=True, pin_memory=True,)

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

        #get current epoch
        current_epoch = self.trainer.current_epoch

        #check if we need to change the dataset
        if current_epoch+1 in self.continual_epochs_schedule and current_epoch > 5:
            self.continual_step += 1
            self.setup_test()

            current_classes = self.continual_classes_schedule[self.continual_step]

            # set weight of everything to 1 and class 2 to 0.0125
            self.weights = torch.ones(current_classes).cuda()
            self.weights[0] = 0.125

            self.criterion = nn.CrossEntropyLoss(weight=self.weights, reduction='none', ignore_index=-100,
                                                label_smoothing=0.1,)

            self.f1 = torchmetrics.F1Score(num_classes=current_classes, task="multiclass", average=None, ).cuda()
            self.f1_micro = torchmetrics.F1Score(num_classes=current_classes, task="multiclass", average='micro', ).cuda()
            
            self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=current_classes, task="multiclass", normalize='true').cuda()

            # Set teacher as a frozen deep copy of the current model
            self.teacher = copy.deepcopy(self.model)
            self.teacher = self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False


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

        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=1e-4, weight_decay=0.01, )  # fused=True)
        # optimizer = Lion(self.trainer.model.parameters(), lr=3e-4/3, weight_decay=0.1*15, use_triton=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                        total_steps=self.trainer.estimated_stepping_batches,
                                                        pct_start=0.3)

        # return optimizer
        return [optimizer], [{'scheduler': scheduler, "interval": "step"}]
