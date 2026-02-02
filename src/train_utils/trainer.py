import os 
import logging 
import copy
from tqdm import tqdm

import numpy as np
import mindspore as ms
from mindspore import nn, ops
from .loss_fn import HingeLoss



class MyTrainOneStepCell(nn.Cell):
    """自定义单步训练Cell"""
    def __init__(self, model, loss_fn, optimizer):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # 获取网络中需要计算梯度的参数
        self.weights = optimizer.parameters
        # 创建梯度计算函数
        self.grad_fn = ops.value_and_grad(self.forward_with_loss, None, self.weights)
    
    def forward_with_loss(self, x, y,weights):
        """前向传播并计算损失"""
        pred = self.model(x)
        loss = self.loss_fn(pred, y, weights)
        return loss
    
    def construct(self, x, y,weights):
        """执行一步训练"""
        # 计算损失和梯度
        loss, grads = self.grad_fn(x, y, weights)
        # 更新参数
        self.optimizer(grads)
        return loss
    
    
    
class Trainer():
    def __init__(self,config, model,weights,train_loader):
        self.config = config
        self.model = model 

        self.train_loader = train_loader
        
        self.weights = weights
        self.best_error = float("inf")
        self.best_model = None

        
        
        self.setup_logging()
        self.setup_dataset_config(config["dataset_config"])
        self.setup_optimization_config(config["train_config"]["optimization_config"])
        self.train_model = MyTrainOneStepCell(model, self.loss_fn, self.optimizer)




        
        
    
    
    
    def setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)


    def setup_dataset_config(self,dataset_config):
        self.n_train_samples = dataset_config["n_train_samples"]

    def setup_optimization_config(self,optimization_config):
        self.n_epochs = optimization_config["n_epochs"]
        self.early_stopping = optimization_config["early_stopping"]
        self.early_stopping_patience = optimization_config["early_stopping_patience"]
        self.early_stopping_delta = optimization_config["early_stopping_delta"]
        self.setup_loss_fn(optimization_config["loss_fn"])
        self.setup_optimizer(optimization_config)

    def setup_loss_fn(self,loss_fn):
        if loss_fn == "hinge":
            self.loss_fn = HingeLoss()
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")


    def setup_optimizer(self,optimization_config):
        optimizer = optimization_config["optimizer"]
        learning_rate = optimization_config["learning_rate"]
        if optimizer == "adam":
            self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = nn.SGD(self.model.trainable_params(), learning_rate=learning_rate)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        



    def _update_best_model(self, error):
        if error < self.best_error:
            self.best_model = copy.deepcopy(self.model)
            # self.best_model = copy.deepcopy(self.model)
            self.best_error = error
            self.early_stopping_counter = 0



    def train_epoch(self,epoch,train_loader):
        total_loss = 0.0
        total_error = 0.0
        
        
        desc = f"Train Epoch {epoch+1}/{self.n_epochs}"
        with tqdm(train_loader, desc=desc, postfix={"loss": 0.0}, leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                data,labels = batch
                # 转换为 MindSpore Tensor
                data = ms.Tensor(data, dtype=ms.float32)
                labels = ms.Tensor(labels, dtype=ms.float32)
                
                batch_start = batch_idx * train_loader.batch_size
                batch_end = min((batch_idx + 1) * train_loader.batch_size, self.n_train_samples)
                batch_weights = ms.Tensor(self.weights[batch_start:batch_end], dtype=ms.float32)

                # 执行一步训练
                loss = self.train_model(data, labels, batch_weights)
                
                predictions = ops.sign(self.model(data).flatten()).asnumpy()
                batch_error = (predictions != labels.asnumpy()) * batch_weights.asnumpy()
                batch_error = batch_error.sum()  # 已经是 numpy 标量
                
                total_loss += loss.asnumpy()
                total_error += batch_error  # 不需要 .asnumpy()
                
   
            
        training_loss = total_loss 
        training_error = total_error 
        
        
        return training_loss, training_error
            
            
        
    def eval_epoch(self,epoch,data_loader):

        total_error = 0.0
        predictions_list = []
        labels_list = []
        
        
        desc = f"Train Epoch {epoch+1}/{self.n_epochs}"
        with tqdm(data_loader, desc=desc, postfix={"loss": 0.0}, leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                data,labels = batch
                # 转换为 MindSpore Tensor
                data = ms.Tensor(data, dtype=ms.float32)
                labels = ms.Tensor(labels, dtype=ms.float32)
                
                batch_start = batch_idx * data_loader.batch_size
                batch_end = min((batch_idx + 1) * data_loader.batch_size, self.n_train_samples)
                batch_weights = ms.Tensor(self.weights[batch_start:batch_end], dtype=ms.float32)

                predictions = ops.sign(self.model(data).flatten()).asnumpy()
                batch_error = (predictions != labels.asnumpy()) * batch_weights.asnumpy()
                batch_error = batch_error.sum()  # 已经是 numpy 标量
                
                total_error += batch_error  # 不需要 .asnumpy()
                predictions_list.append(predictions)
                labels_list.append(labels)
                
   
            
        eval_error = total_error 
        
        
        return eval_error,np.concatenate(predictions_list),np.concatenate(labels_list)
    
    
    
    def train(self):
        for epoch in tqdm(range(self.n_epochs),desc="Training",leave=True):
            training_loss,training_error = self.train_epoch(epoch,self.train_loader)
            train_error,train_predictions,train_labels = self.eval_epoch(epoch,self.train_loader)  
            
            
            
            self._update_best_model(train_error)
            
            
            if self.early_stopping:
                self.early_stopping_counter += 1 
                if self.early_stopping_counter >= self.early_stopping_patience and self.best_error <= self.early_stopping_delta: 
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1} with best error {self.best_error}")
                    break
                
        return self.best_model,train_error,train_predictions,train_labels