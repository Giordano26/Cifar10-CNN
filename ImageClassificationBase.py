# Contains helper methods for training & validation.
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassificationBase(nn.Module):
    def _accuracy(self,outputs, labels):
        # Ignores the value of max tensor, just cares about the index
        _, preds = torch.max(outputs, dim=1) 
        # From the index calculates a boolean (correct/incorrect) indicating if the presumptions were right among the len(preds)
        acc_result = torch.tensor(torch.sum(preds == labels).item() / len(preds)) 
        return acc_result
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Passes images to the inheritance from nn.Module to generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss with cross entropy (sigmoid and softmax)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Passes images to the inheritance from nn.Module to generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss with cross entropy (sigmoid and softmax)
        acc = self._accuracy(out, labels)           # Calculate accuracy from our method
        
        results = {
         'val_loss': loss.detach(), 
         'val_acc': acc
        }
        
        return results
        
    #Returns stats from each epoch
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        
        results = {
            'val_loss': epoch_loss.item(), 
            'val_acc': epoch_acc.item()
        }
        
        return results
    
    #Print the stats
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
