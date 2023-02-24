import torch
from timm.utils import accuracy

from utils import map_labels
from torch.nn.parallel import DistributedDataParallel as DDP
import utils.logger as logger

class Trainer:
    def __init__(self, 
                model,
                lr_scheduler,
                optimizer,
                data_loader_train,
                data_loader_val,
                global_labels_val,
                gpu_id) -> None:
        self.model = model.to(gpu_id)
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.loss_function = torch.nn.CrossEntropyLoss()

        self.gpu_id = gpu_id
        self.model = DDP(model, device_ids=[gpu_id])
        
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.global_labels_val = global_labels_val

    def train(self,
              epochs: int
              ) -> None:
        max_accuracy = 0

        for epoch in range(epochs):
            train_stats = self.train_one_epoch(epoch)

            self.lr_scheduler.step(epoch)

            evaluation_stats = self.evaluate(eval=False)

            print(f"Accuracy on validation dataset: {evaluation_stats['acc']:.2f}% ± {evaluation_stats['confidence_interval']:.4f}%")

            max_accuracy = max(max_accuracy, evaluation_stats['acc'])
            print(f'Max accuracy: {max_accuracy:.2f}%')


    def train_one_epoch(self,
                        epoch) -> dict:
        
        
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('n_ways', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('n_imgs', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
        header = 'Epoch: [{}]'.format(epoch)

        self.data_loader_train.sampler.set_epoch(epoch)
        self.model.train()
        
        self.data_loader_train.sampler.set_epoch(epoch)
        for batch in metric_logger.log_every(self.data_loader_train, 10, header):
            SupportTensor, SupportLabel, x, y, _ = batch
            SupportTensor = SupportTensor.to(self.gpu_id)
            SupportLabel = SupportLabel.to(self.gpu_id)
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)
            
            # Forward 
            with torch.cuda.amp.autocast():
                logits = self.model(query=x, support=SupportTensor, support_labels=SupportLabel)
            
            logits = torch.squeeze(logits)
            y = y.view(-1)
            loss = self.loss_function(logits, y)
            loss_value = loss.item()
            
            self.optimizer.zero_grad()
            self.optimizer.step()
            self.lr_scheduler.step()
                    
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=lr)
            metric_logger.update(n_ways=SupportLabel.max()+1)
            metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])
        
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                

    @torch.no_grad()
    def evaluate(self, 
                eval=False):
        data_loader = self.data_loader_val if eval else self.data_loader_val
        gloal_label_id = self.global_labels_val if eval else self.global_labels_val

        # Logger 
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter('n_ways', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('n_imgs', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('acc', logger.SmoothedValue(window_size=len(data_loader.dataset)))
        header = 'Test:'
        
        all_acc = []
        y_pred = []
        y_target = []

        self.model.eval()
        
        for ii, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
            SupportTensor, SupportLabel, x, y, label_to_class = batch
            SupportTensor = SupportTensor.to(self.gpu_id)
            SupportLabel = SupportLabel.to(self.gpu_id)
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)

            with torch.cuda.amp.autocast():
                logits = self.model(query=x, support=SupportTensor, support_labels=SupportLabel)
                
            logits = torch.squeeze(logits)
            y = y.view(-1)
            pred = torch.argmax(logits, dim=-1)
            
            # Calculate accuracy
            acc = accuracy(logits, y)[0]
            all_acc.append(acc.item())
            
            # Loss 
            loss = self.loss_function(logits, y)
            
            # Map classified labels to global labels
            y = map_labels(gloal_label_id,label_to_class, y)
            pred = map_labels(gloal_label_id,label_to_class, pred)
            
            # Append results 
            y_pred.extend(pred)
            y_target.extend(y)
            
            # Create Logger
            batch_size = x.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc'].update(acc.item(), n=batch_size)
            metric_logger.update(n_ways=SupportLabel.max()+1)
            metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])
        
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top.global_avg:.3f} ± {top.mean_confidence_interval: .4f} loss {losses.global_avg:.3f}'
            .format(top=metric_logger.acc, losses=metric_logger.loss))
        
        return_dict = {}
        return_dict['acc'] = metric_logger.meters['acc'].avg
        return_dict['confidence_interval'] = metric_logger.meters['acc'].mean_confidence_interval
        return_dict['y_pred'] = y_pred
        return_dict['y_target'] = y_target
            
        return return_dict
        
        
        