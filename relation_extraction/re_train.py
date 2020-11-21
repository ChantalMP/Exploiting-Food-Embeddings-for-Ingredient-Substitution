'''
Trains FoodBERT for relation extraction using the comment data
'''
from random import uniform

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from relation_extraction.helpers.re_dataset import REDataset
from relation_extraction.helpers.re_model import ReFoodBERT


class TrainWrapper():
    def __init__(self):
        self.epochs = 10
        self.batchsize = 32
        self.lr = 2e-5
        num_workers = 1
        self.continue_training = False
        self.writer: SummaryWriter = SummaryWriter()

        self.dataset = REDataset(subset='train')
        weights = self.compute_class_weights(self.dataset)
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights))
        self.dataloader = DataLoader(self.dataset, batch_size=self.batchsize, sampler=sampler, num_workers=num_workers)

        self.dataset_val = REDataset(subset='val')
        weights = self.compute_class_weights(self.dataset_val)
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights))
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=self.batchsize, sampler=sampler, num_workers=num_workers)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model: ReFoodBERT = ReFoodBERT(device=self.device)
        self.model.to(self.device)

        self.opt = AdamW(self.model.parameters(), lr=self.lr)
        total_steps = self.epochs * len(self.dataloader)
        self.scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=0, num_training_steps=total_steps)

    def compute_class_weights(self, dataset):
        def assign_bucket_idx_to_label(label):
            if label[0] <= 0 and label[1] <= 0:
                return 0
            elif label[0] == 1 and label[1] != 1:
                return 1
            elif label[0] != 1 and label[1] == 1:
                return 2
            else:  # 1,1
                if uniform(0, 1) >= 0.5:
                    return 1
                else:
                    return 2

        n_classes = 3
        count = [0] * n_classes  # We have three buckets: 0, ->, <-
        for label in tqdm(dataset.labels, total=len(dataset), desc='Calcuting Class Counts'):
            class_idx = assign_bucket_idx_to_label(label)
            count[class_idx] += 1

        weight_per_class = [0.] * n_classes
        for i in range(n_classes):
            weight_per_class[i] = 1 / float(count[i])
        weight = [0] * len(dataset)
        for idx, label in enumerate(tqdm(dataset.labels, total=len(dataset), desc='Assigning Weights')):
            class_idx = assign_bucket_idx_to_label(label)
            weight[idx] = weight_per_class[class_idx]
        return weight

    @torch.no_grad()
    def validate(self, global_step):
        total_val_accuracy = 0.
        for sentences, labels in tqdm(self.dataloader_val, total=len(self.dataloader_val),
                                      desc="Validation - Sentences", mininterval=60):
            # Perform a forward pass
            self.model.eval()
            labels = labels.to(self.device)
            _, logits = self.model(sentences=sentences,
                                   labels=labels)

            preds = (torch.sigmoid(logits) >= 0.5)
            labels = labels.flatten()
            known_mask = (labels != -1)

            total_val_accuracy += (torch.sum(preds[known_mask] == labels[known_mask]).float() / len(labels[known_mask])).item()

        # Calculate the average loss over all of the batches.
        avg_val_accuracy = total_val_accuracy / len(self.dataloader_val)

        self.writer.add_scalar('Accuracy/Val', avg_val_accuracy, global_step=global_step)

        return avg_val_accuracy

    def train(self):
        best_val_acc = -1

        epoch = 0
        checkpoint_path = f"relation_extraction/models/RE_checkpoint.pt"
        model_path = f"relation_extraction/models/RE_model_filtered.pt"
        if self.continue_training:
            checkpoint = torch.load(checkpoint_path)
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.load_state_dict(torch.load(model_path))
            epoch = checkpoint['epoch'] + 1

        # For each epoch...
        global_step = 0
        for epoch_idx in tqdm(range(epoch, self.epochs), total=self.epochs, desc="Epochs", leave=True):
            total_train_loss = 0.
            total_train_accuracy = 0.
            total_count = 0.
            for idx, (sentences, labels) in enumerate(tqdm(self.dataloader, total=len(self.dataloader), desc="Sentences", mininterval=60)):
                self.model.train()
                labels = labels.to(self.device)
                # Perform a forward pass
                loss, logits = self.model(sentences=sentences, labels=labels)

                preds = (torch.sigmoid(logits) >= 0.5)
                labels = labels.flatten()
                known_mask = (labels != -1)
                total_train_loss += loss.item()
                loss.backward()
                total_count += 1
                total_train_accuracy += (torch.sum(preds[known_mask] == labels[known_mask]).float() / len(labels[known_mask])).item()
                self.opt.step()
                self.scheduler.step()
                self.model.zero_grad()

                if idx % 500 == 0:
                    # Calculate the average loss over all of the batches.
                    avg_train_loss = total_train_loss / total_count
                    avg_train_accuracy = total_train_accuracy / total_count

                    print(f'\nLoss: {avg_train_loss:.4f} Acc: {avg_train_accuracy}\n')
                    self.writer.add_scalar('Loss/Train', avg_train_loss, global_step=global_step)
                    self.writer.add_scalar('Accuracy/Train', avg_train_accuracy, global_step=global_step)
                    # Validation
                    val_acc = self.validate(global_step=global_step)
                    print(f'Val Acc: {val_acc}')
                    if val_acc > best_val_acc:
                        print(f"Saving model with accuracy {val_acc}")
                        torch.save(self.model.state_dict(),
                                   f"relation_extraction/models/RE_model_best_filtered.pt")
                        best_val_acc = val_acc

                    avg_train_loss = 0.
                    avg_train_accuracy = 0.
                    total_train_loss = 0.
                    total_train_accuracy = 0.
                    total_count = 0.
                    global_step += 1

            # Save for continuing training
            torch.save(self.model.state_dict(), model_path)
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': self.opt.state_dict()
            }, checkpoint_path)


if __name__ == '__main__':
    trainer = TrainWrapper()
    trainer.train()
