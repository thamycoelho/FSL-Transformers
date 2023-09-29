import torch
import json

from timm.utils import accuracy

from utils import map_labels, generate_confusion_matrix, get_aggregator
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as nn
import pandas as pd
import utils.logger as logger
from model import generate_prototype

class Trainer:
    def __init__(self, 
                model,
                lr_scheduler,
                optimizer,
                data_loader_train,
                data_loader_val,
                global_labels_val,
                device,
                output_dir, 
                experiment_name = None) -> None:
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.loss_function = torch.nn.CrossEntropyLoss()

        self.device = device
        
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.global_labels_val = global_labels_val
        self.output_dir = output_dir
        self.experiment_name = experiment_name

    def train(self,
              epochs: int,
              args
              ) -> None:
        max_accuracy = (0, 0) if not args.max_acc else args.max_acc

        for epoch in range(args.start_epoch, epochs):
            train_stats = self.train_one_epoch(epoch)

            evaluation_stats = self.evaluate(eval=False, record_wandb=args.wandb)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in evaluation_stats.items()},
                     'epoch': epoch}

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth', self.output_dir / 'best.pth']
                for checkpoint_path in checkpoint_paths:
                    state_dict = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                        'epoch': epoch,
                        'max_acc': max_accuracy,
                        'args': args,
                    }
                    
                    torch.save(state_dict, checkpoint_path)
                    if evaluation_stats["acc"] <= max_accuracy[0]:
                        break # do not save best.pth
                
            print(f"Accuracy on validation dataset: {evaluation_stats['acc']:.2f}% ± {evaluation_stats['confidence_interval']:.4f}%")

            max_accuracy = max(max_accuracy, (evaluation_stats['acc'], evaluation_stats['confidence_interval']), key= lambda x: x[0])
            print(f'Max accuracy: {max_accuracy[0]:.2f}% ± {max_accuracy[1]:.4f}%')

            if self.output_dir:
                log_stats['best_test_acc'] = max_accuracy[0]
                log_stats['confidence_interval'] = max_accuracy[1]
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")


    def train_one_epoch(self,
                        epoch) -> dict:
        
        
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('n_ways', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('n_imgs', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
        header = 'Epoch: [{}]'.format(epoch)

        self.model.train()
        
        for batch in metric_logger.log_every(self.data_loader_train, 10, header):
            SupportTensor, SupportLabel, x, y, _ = batch
            SupportTensor = SupportTensor.to(self.device)
            SupportLabel = SupportLabel.to(self.device)
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward 
            with torch.cuda.amp.autocast():
                logits = self.model(query=x, support=SupportTensor, support_labels=SupportLabel)
            
            logits = torch.squeeze(logits)
            y = y.view(-1)
            loss = self.loss_function(logits, y)
            loss_value = loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)
                    
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
                eval=False,
                resume=False,
                record_wandb=True):
        data_loader = self.data_loader_val 
        global_label_id = self.global_labels_val 

        # Logger 
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter('n_ways', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('n_imgs', logger.SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('acc', logger.SmoothedValue(window_size=len(data_loader.dataset)))
        header = 'Test:'
        
        all_acc = []
        y_pred = []
        y_target = []

        df = pd.DataFrame(columns=['Image File', 'Label'])
        self.model.eval()
        
        for ii, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
            SupportTensor, SupportLabel, x, y, support_files, query_files, label_to_class = batch
            SupportTensor = SupportTensor.to(self.device)
            SupportLabel = SupportLabel.to(self.device)
            x = x.to(self.device)
            y = y.to(self.device)

            label = []
            predicted = []
            files = []
            df_episode = pd.DataFrame()
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
        
            # Append results 
            label.extend([label_to_class[x.item()][0] for x in y])
            predicted.extend([label_to_class[x.item()][0] if query_files[i] not in support_files else None for i, x in enumerate(pred)])
            files.extend([x[0] for x in query_files])

            # Append results 
            y_pred.extend(pred)
            y_target.extend(y)
            
            # Create Logger
            batch_size = x.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc'].update(acc.item(), n=batch_size)
            metric_logger.update(n_ways=SupportLabel.max()+1)
            metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])

            df_episode['Image File'] = files
            df_episode['Label'] = label
            df_episode['Epipsode {}'.format(ii)] = predicted
            
            df = pd.merge(df, df_episode, how='outer', on=['Image File', 'Label'])
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top.global_avg:.3f} ± {top.mean_confidence_interval: .4f} loss {losses.global_avg:.3f}'
            .format(top=metric_logger.acc, losses=metric_logger.loss))            

        return_dict = {}
        return_dict['acc'] = metric_logger.meters['acc'].avg
        return_dict['confidence_interval'] = metric_logger.meters['acc'].mean_confidence_interval

        if self.output_dir:
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(return_dict) + "\n")

            df.to_csv(self.output_dir / 'experiment.csv', index=False, sep=',')

        return return_dict
        
        
    @torch.no_grad()
    def extract_features(self, args):
        data_loader = self.data_loader_val 

        features_by_class = {}
        for ii, batch in enumerate(data_loader):
            x, y, img_file, label_to_class = batch

            x = x.to(self.device)

            with torch.cuda.amp.autocast():
                features = self.model.get_features(batch=x)
            
            for feat_idx in range(features.shape[0]):
                label = y[feat_idx]

                cls = label_to_class[label.item()][0]

                if not cls in features_by_class:
                    features_by_class[cls] = []

                features_by_class[cls].append((features[feat_idx], img_file[feat_idx]))


        return features_by_class
    
    def classify_general_(self, args):
        # Logger 
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter('acc', logger.SmoothedValue(window_size=len(self.data_loader_train.dataset)))
        header = 'Classify:'

        support = self.data_loader_train
        query = self.data_loader_val
        global_label_id = self.global_labels_val 
        
        y_pred = []
        y_target = []

        df = pd.DataFrame()

        self.model.eval()

        for episode, support_batch in enumerate(support):
            print(f'Episode {episode}:')
            SupportTensor, SupportLabel, sup_image_file, sup_label_to_class = support_batch
            SupportTensor = SupportTensor.to(self.device)
            SupportLabel = SupportLabel.to(self.device)

            SupportTensor = torch.squeeze(SupportTensor)
            label = []
            predicted = []
            files = []
            correct = 0

            sup_image_file = [x[0] for x in sup_image_file]

            for n, query_batch in enumerate(query):
                QueryTensor, QueryLabel, img_file, query_label_to_class = query_batch
                QueryTensor = QueryTensor.to(self.device)
                QueryLabel = QueryLabel.to(self.device)

                aggregator = get_aggregator(args.aggregator, QueryTensor)
                prototype = generate_prototype(SupportLabel, SupportLabel.max()+1, SupportTensor, aggregator, args.aggregator)
                logits = self.model(prototype, QueryTensor)

                logits = torch.squeeze(logits)
                QueryLabel = QueryLabel.view(-1)
                pred = torch.argmax(logits, dim=-1)

                # Calculate accuracy                
                correct += (pred == QueryLabel).sum()

                # Map classified labels to global labels
                QueryLabel = map_labels(global_label_id,query_label_to_class, QueryLabel)
                pred = map_labels(global_label_id,sup_label_to_class, pred)
                # Append results 
                label.extend([sup_label_to_class[x][0] for x in QueryLabel])
                predicted.extend([sup_label_to_class[x][0] if img_file[i] not in sup_image_file else None for i, x in enumerate(pred)])
                files.extend([x for x in img_file])

                y_pred.extend(pred)
                y_target.extend(QueryLabel)
                

            if episode == 0:
                df['Image File'] = files
                df['Label'] = label
            df['Epipsode {}'.format(episode)] = predicted
            acc = (correct.item() / len(predicted)) * 100
            print("Accuracy:", acc)

            metric_logger.meters['acc'].update(acc)

        print('* Acc@1 {top.global_avg:.3f} ± {top.mean_confidence_interval: .4f}'
        .format(top=metric_logger.acc))

        return df
    
    def classify_fsl_(self, args):
        dataloader = self.data_loader_val
        global_label_id = self.global_labels_val 
        
        # Logger 
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter('acc', logger.SmoothedValue(window_size=len(dataloader.dataset)))
        header = 'Classify:'

        
        y_pred = []
        y_target = []

        df = pd.DataFrame(columns=['Image File', 'Label'])

        self.model.eval()

        for episode, batch in enumerate(dataloader):
            SupportTensor, SupportLabel, QueryTensor, QueryLabel, support_files, query_files, label_to_class = batch
            SupportTensor = SupportTensor.to(self.device)
            SupportLabel = SupportLabel.to(self.device)
            QueryTensor = QueryTensor.to(self.device)
            QueryLabel = QueryLabel.to(self.device)

            SupportTensor = torch.squeeze(SupportTensor)

            label = []
            predicted = []
            files = []
            df_episode = pd.DataFrame()

            aggregator = get_aggregator(args.aggregator, QueryTensor)
            prototype = generate_prototype(SupportLabel, SupportLabel.max()+1, SupportTensor, aggregator, args.aggregator)
            logits = self.model(prototype, QueryTensor)

            logits = torch.squeeze(logits)
            QueryLabel = QueryLabel.view(-1)
            pred = torch.argmax(logits, dim=-1)

            # Map classified labels to global labels
            QueryLabel = map_labels(global_label_id, label_to_class, QueryLabel)
            pred = map_labels(global_label_id, label_to_class, pred)
            
            # Append results 
            label.extend([label_to_class[x][0] for x in QueryLabel])
            predicted.extend([label_to_class[x][0] if query_files[i] not in support_files else None for i, x in enumerate(pred)])
            files.extend([x[0] for x in query_files])

            y_pred.extend(pred)
            y_target.extend(QueryLabel)

            df_episode['Image File'] = files
            df_episode['Label'] = label
            df_episode['Epipsode {}'.format(episode)] = predicted
            
            df = pd.merge(df, df_episode, how='outer', on=['Image File', 'Label'])
        return df


    def classify_from_features(self, args):
        if args.eval_general:
            df = self.classify_general_(args)
        elif args.eval_fsl:
            df = self.classify_fsl_(args)

        return df
        
        