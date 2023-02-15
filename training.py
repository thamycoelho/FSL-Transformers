import torch
from timm.utils import accuracy

from utils import map_labels, to_device
import utils.logger as logger

def train_one_epoch(data_loader,
                    model,
                    loss_func,
                    lr_scheduler,
                    optimizer,
                    epoch,
                    device):
    
    
    metric_logger = logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('n_ways', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    model.train()
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        batch = to_device(batch, device)
        SupportTensor, SupportLabel, x, y, _ = batch
        
        # Forward 
        with torch.cuda.amp.autocast():
            logits = model(query=x, support=SupportTensor, support_labels=SupportLabel)
        
        logits = torch.squeeze(logits)
        y = y.view(-1)
        loss = loss_func(output, y)
        loss_value = loss.item()
        
        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()
                
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
        metric_logger.update(n_ways=SupportLabel.max()+1)
        metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
              

@torch.no_grad()
def evaluate(data_loader, model, loss_function, device, gloal_label_id):
    # Logger 
    metric_logger = logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('n_ways', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('acc', logger.SmoothedValue(window_size=len(data_loader.dataset)))
    header = 'Test:'
    
    all_acc = []
    y_pred = []
    y_target = []
    
    model.eval()
    
    for ii, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        batch = to_device(batch, device)
        SupportTensor, SupportLabel, x, y, label_to_class = batch
        
        with torch.cuda.amp.autocast():
            logits = model(query=x, support=SupportTensor, support_labels=SupportLabel)
            
        logits = torch.squeeze(logits)
        y = y.view(-1)
        pred = torch.argmax(logits, dim=-1)
        
        # Calculate accuracy
        acc = accuracy(logits, y)[0]
        all_acc.append(acc.item())
        
        # Loss 
        loss = loss_function(logits, y)
        
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
        
        
        