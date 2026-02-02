
import pickle
from tqdm import tqdm
import numpy as np
import copy


class QAdaBoost:
    def __init__(self,n_train):
        """
        初始化QAdaBoost二分类器

        参数:
        - n_estimators: 弱分类器的数量
        """
        self.n_estimators = 0
        self.models = []
        self.alphas = []
        self.errors = []
        self.Zs = []
        self.weights = np.ones(n_train) / n_train
        self.weights_history = []
        self.weights_history.append(self.weights)
        self.gamma = None


 
    def __call__(self, x, n_estimators=None):
        if n_estimators is None:
            n_estimators = self.n_estimators
        sum_pred = 0
        for i in range(n_estimators):
            predictions = self.get_predictions_i(x,i)
            sum_pred += self.alphas[i] * predictions
        return np.sign(sum_pred)

    def get_predictions_i(self,x,i):
        predictions = self.models[i](x)
        return np.sign(predictions)


    def record(self, base_model, error):
        """
        记录基分类器和对应的alpha值以及错误率

        参数:
        - base_model: 基分类器
        - alpha: 基分类器的权重
        - error: 基分类器的错误率
        """
        self.n_estimators += 1
        self.models.append(base_model)
        self.errors.append(error)
        if error != 0:
            alpha_t = 1/2 * np.log((1 - error) / error)
            self.alphas.append(alpha_t)
        else:
            self.alphas.append(1)
        print(f"Recorded estimator {len(self.models)} with error {error:.4f}")


    def get_weights(self):
        return self.weights
    
    def update_weights(self,y_train,predictions):
        """
        更新样本权重

        参数:
        - y_train: 训练标签
        - predictions: 预测值

        返回:
        - new_weights: 更新后的样本权重
        """
        Z_t = 2 * np.sqrt(self.errors[self.n_estimators-1] * (1 - self.errors[self.n_estimators-1]))
        weights_t = (self.weights * np.exp(-self.alphas[self.n_estimators-1] * y_train * predictions)) / Z_t
        print("sum of weights: ", np.sum(weights_t))
        self.Zs.append(Z_t)
        self.weights = weights_t
        self.weights_history.append(weights_t)
        
        return weights_t
    
    
    def get_predictions(self,data_loader,n_estimators=None):
        predictions = []
        y_train = []
        for batch in tqdm(data_loader, desc="Getting predictions"):
            image, labels = batch
            if n_estimators is not None:
                outputs = self(image,n_estimators)
            else:
                outputs = self(image)
            predictions.append(np.sign(outputs))
            y_train.append(labels)
        return np.concatenate(predictions),np.concatenate(y_train)

    def get_error_bound(self,tight=False,n_estimators=None):
        """
        获取错误率上界

        参数:
        - tight: 是否使用紧上界
        """ 
        if n_estimators is None:
            n_estimators = self.n_estimators
        if tight:
            return np.exp(-2 * np.sum( ( 1/2 - np.array(self.errors[:n_estimators]))**2) )
        else:
            gamma = np.min(1/2 - np.array(self.errors[:n_estimators]))
            self.gamma = gamma
            return np.exp(- 2 * gamma**2 * n_estimators)


    
