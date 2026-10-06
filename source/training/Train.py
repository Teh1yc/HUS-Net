from source.utils import accuracy, TotalMeter, count_params, isfloat, BCEWithLogitsLossL2
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import wandb
from omegaconf import DictConfig
from typing import List, Dict
import torch.utils.data as utils
from source.components import lr_scheduler_factory, LRScheduler
import logging
from scipy.special import expit
import ipdb


class Train:
    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader]) -> None:

        self.all_saved_x = []
        self.evidence_records = []  # 用于记录证据信息

        self.cfg = cfg
        self.device = self.cfg.device
        self.fold_idx = 0  # 默认值，实际由主函数传入
        self.is_cross_validation = False  # 是否进行交叉验证

        self.logger = logging.getLogger()
        self.model = model.to(self.device)
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = BCEWithLogitsLossL2(self.model, self.cfg.training.l2)

        self.best_test_metrics = {
            "accuracy": 0.0,
            "auc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        
        # 用于保存折结果的变量
        self.fold_results = {
            "fold_idx": self.fold_idx,
            "best_epoch": 0,
            "best_metrics": None,
            "final_metrics": None,
        }

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy, self.val_accuracy, self.test_accuracy = [
            TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.test_accuracy, self.train_loss, self.test_loss]:
            meter.reset()

    def set_fold_info(self, fold_idx, total_folds=None):
        """设置当前折信息，由主函数调用"""
        self.fold_idx = fold_idx
        self.is_cross_validation = True
        if total_folds:
            self.logger.info(f'Starting Fold {self.fold_idx + 1}/{total_folds}')
        else:
            self.logger.info(f'Starting Fold {self.fold_idx + 1}')
        self.fold_results["fold_idx"] = fold_idx

    def train_per_epoch(self, epoch, optimizer, lr_scheduler):
        self.model.train()
        self.evidence_records = []  # 每个epoch重置记录

        for iteration, data in enumerate(self.train_dataloader):
            optimizer.zero_grad()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            data = data.to(self.device)
            label = data.y.to(self.device)

            # 模型前向传播（返回预测和损失）
            output = self.model(
                data, epoch=epoch, iteration=iteration, test_phase=False
            )
            
            if isinstance(output, tuple) and len(output) == 2:
                predict, loss = output
            else:
                # 兼容旧版本
                predict = output[0] if isinstance(output, tuple) else output
                loss = output[1] if isinstance(output, tuple) and len(output) > 1 else self.loss_fn(predict, label)

            predict = predict.squeeze()
            
            # Ensure that the output tensor has at least one dimension
            if predict.dim() == 0:
                predict = predict.unsqueeze(0)
            
            loss.backward(retain_graph=True)
            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.step()
            
            acc = accuracy(predict, label)
            self.train_accuracy.update_with_weight(acc, label.shape[0])

            # 可选：记录证据信息用于监控
            if iteration % 50 == 0 and hasattr(self.model, 'evidential_selection'):
                # 这里可以记录一些统计信息
                pass

    def test_per_epoch(self, epoch, dataloader, loss_meter, acc_meter):
        labels = []
        logits = []
        all_probs = []  # 记录相关概率
        all_uncertainties = []  # 记录不确定性
        all_evidence = []  # 记录证据强度

        self.model.eval()

        with torch.no_grad():
            for iteration, data in enumerate(dataloader):
                data = data.to(self.device)
                label = data.y.to(self.device)

                # 模型前向传播（测试时返回三个值）
                output = self.model(
                    data, epoch=epoch, iteration=iteration, test_phase=True
                )

                if isinstance(output, tuple) and len(output) == 3:
                    predict, _, evidence_info = output
                elif isinstance(output, tuple) and len(output) == 2:
                    predict, _ = output
                    evidence_info = None
                else:
                    predict = output
                    evidence_info = None

                predict = predict.squeeze()
                
                if predict.dim() == 0:
                    predict = predict.unsqueeze(0)

                # 使用标准损失函数计算测试损失
                loss = self.loss_fn(predict, label)

                loss_meter.update_with_weight(loss.item(), label.shape[0])
                acc = accuracy(predict, label)
                acc_meter.update_with_weight(acc, label.shape[0])
                
                logits += predict.tolist()
                labels += label.tolist()


        # 转换logits为概率和预测
        probabilities = expit(np.array(logits))
        predictions = np.round(probabilities)
        labels = np.array(labels)

        # 计算AUC
        auc = roc_auc_score(labels, probabilities)
        # labels_flipped = labels
        # 计算其他指标
        metric = precision_recall_fscore_support(labels, predictions, average='binary')
        # metric = precision_recall_fscore_support(labels, predictions, average='micro')
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']

        # 记录证据统计信息（如果有）
        # if len(all_probs) > 0:
        #     self.logger.info(f"Evidence Stats - Avg Prob: {np.mean(all_probs):.3f}, "
        #                    f"Avg Uncertainty: {np.mean(all_uncertainties):.3f}, "
        #                    f"Avg Evidence: {np.mean(all_evidence):.1f}")
        
        # 返回详细指标
        detailed_metrics = {
            "auc": auc,
            "accuracy": acc_meter.avg,
            "precision": metric[0],
            "recall": metric[1],
            "f1": metric[2],
            "sensitivity": recall[1],
            "specificity": recall[0],
            "loss": loss_meter.avg,
        }
        
        return detailed_metrics

    def train(self):
        # 根据是否交叉验证显示不同的标题
        if self.is_cross_validation:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting training for Fold {self.fold_idx + 1}")
            self.logger.info(f"{'='*60}")
        else:
            print("\nStarting training...")
            self.logger.info(f"\n{'='*60}")
            self.logger.info("Starting training")
            self.logger.info(f"{'='*60}")
        
        self.current_step = 0

        for epoch in range(self.epochs):
            self.reset_meters()
            
            # 训练一个epoch
            self.train_per_epoch(epoch, self.optimizers[0], self.lr_schedulers[0])
            
            # 测试
            test_metrics = self.test_per_epoch(
                epoch, self.test_dataloader, self.test_loss, self.test_accuracy
            )

            # 更新最佳指标
            if self.test_accuracy.avg > self.best_test_metrics["accuracy"]:
                self.best_test_metrics["accuracy"] = self.test_accuracy.avg
                self.best_test_metrics["auc"] = test_metrics["auc"]
                self.best_test_metrics["sensitivity"] = test_metrics["sensitivity"]
                self.best_test_metrics["specificity"] = test_metrics["specificity"]
                self.best_test_metrics["precision"] = test_metrics["precision"]
                self.best_test_metrics["recall"] = test_metrics["recall"]
                self.best_test_metrics["f1"] = test_metrics["f1"]
                self.fold_results["best_epoch"] = epoch + 1
                
                # 保存最佳epoch的完整指标
                self.fold_results["best_metrics"] = {
                    "epoch": epoch + 1,
                    "train_loss": self.train_loss.avg,
                    "train_accuracy": self.train_accuracy.avg,
                    "test_loss": self.test_loss.avg,
                    "test_accuracy": self.test_accuracy.avg,
                    **test_metrics
                }

                # 保存最佳模型
                if self.cfg.model.model_save:
                    if self.is_cross_validation:
                        model_dir = Path("saved_models")
                        model_dir.mkdir(exist_ok=True)
                        model_path = model_dir / f"best_model_fold_{self.fold_idx + 1}.pt"
                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Saved best model for fold {self.fold_idx + 1} with accuracy: {self.test_accuracy.avg:.3f}%")
                    else:
                        torch.save(self.model.state_dict(), 'best_model_evidence.pt')
                        self.logger.info(f"Saved best model with accuracy: {self.test_accuracy.avg:.3f}%")

            # 根据是否交叉验证显示不同的日志前缀
            if self.is_cross_validation:
                # 交叉验证日志格式
                self.logger.info(" | ".join([
                    f'Fold[{self.fold_idx + 1}] Epoch[{epoch+1}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg:.3f}',
                    f'Train Accuracy:{self.train_accuracy.avg:.3f}%',
                    f'Test Loss:{self.test_loss.avg:.3f}',
                    f'Test Accuracy:{self.test_accuracy.avg:.3f}%',
                    f'Test AUC:{test_metrics["auc"]:.4f}',
                    f'Test Sen:{test_metrics["sensitivity"]:.4f}',
                    f'Test Spe:{test_metrics["specificity"]:.4f}',
                    f'Test Prec:{test_metrics["precision"]:.4f}',
                    f'Test F1:{test_metrics["f1"]:.4f}',
                    f'Best Acc:{self.best_test_metrics["accuracy"]:.3f}%',
                    f'Best AUC:{self.best_test_metrics["auc"]:.4f}',
                    f'LR:{self.lr_schedulers[0].lr:.7f}'
                ]))
            else:
                # 非交叉验证日志格式
                self.logger.info(" | ".join([
                    f'Epoch[{epoch+1}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg:.3f}',
                    f'Train Accuracy:{self.train_accuracy.avg:.3f}%',
                    f'Test Loss:{self.test_loss.avg:.3f}',
                    f'Test Accuracy:{self.test_accuracy.avg:.3f}%',
                    f'Test AUC:{test_metrics["auc"]:.4f}',
                    f'Test Sen:{test_metrics["sensitivity"]:.4f}',
                    f'Test Spe:{test_metrics["specificity"]:.4f}',
                    f'Test Prec:{test_metrics["precision"]:.4f}',
                    f'Test F1:{test_metrics["f1"]:.4f}',
                    f'Best Acc:{self.best_test_metrics["accuracy"]:.3f}%',
                    f'Best AUC:{self.best_test_metrics["auc"]:.4f}',
                    f'LR:{self.lr_schedulers[0].lr:.7f}'
                ]))

            # Wandb日志
            if self.cfg.is_wandb:
                if self.is_cross_validation:
                    wandb.log({
                        f"Fold_{self.fold_idx + 1}_Train_Loss": self.train_loss.avg,
                        f"Fold_{self.fold_idx + 1}_Train_Accuracy": self.train_accuracy.avg,
                        f"Fold_{self.fold_idx + 1}_Test_Loss": self.test_loss.avg,
                        f"Fold_{self.fold_idx + 1}_Test_Accuracy": self.test_accuracy.avg,
                        f"Fold_{self.fold_idx + 1}_Test_AUC": test_metrics["auc"],
                        f"Fold_{self.fold_idx + 1}_Test_Sensitivity": test_metrics["sensitivity"],
                        f"Fold_{self.fold_idx + 1}_Test_Specificity": test_metrics["specificity"],
                        f"Fold_{self.fold_idx + 1}_Test_Precision": test_metrics["precision"],
                        f"Fold_{self.fold_idx + 1}_Test_Recall": test_metrics["recall"],
                        f"Fold_{self.fold_idx + 1}_Test_F1": test_metrics["f1"],
                        f"Fold_{self.fold_idx + 1}_Best_Accuracy": self.best_test_metrics["accuracy"],
                        f"Fold_{self.fold_idx + 1}_Best_AUC": self.best_test_metrics["auc"],
                    })
                else:
                    wandb.log({
                        "Train Loss": self.train_loss.avg,
                        "Train Accuracy": self.train_accuracy.avg,
                        "Test Loss": self.test_loss.avg,
                        "Test Accuracy": self.test_accuracy.avg,
                        "Test AUC": test_metrics["auc"],
                        "Test Sensitivity": test_metrics["sensitivity"],
                        "Test Specificity": test_metrics["specificity"],
                        "Test Precision": test_metrics["precision"],
                        "Test Recall": test_metrics["recall"],
                        "Test F1": test_metrics["f1"],
                        "Best Test Accuracy": self.best_test_metrics["accuracy"],
                        "Best Test AUC": self.best_test_metrics["auc"],
                    })

        # 保存最终epoch的结果
        self.fold_results["final_metrics"] = {
            "epoch": self.epochs,
            "train_loss": self.train_loss.avg,
            "train_accuracy": self.train_accuracy.avg,
            "test_loss": self.test_loss.avg,
            "test_accuracy": self.test_accuracy.avg,
            **test_metrics
        }
        
        # 打印本折最终结果
        if self.is_cross_validation:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Fold {self.fold_idx + 1} Final Results:")
            self.logger.info(f"  Best Epoch: {self.fold_results['best_epoch']}")
            self.logger.info(f"  Best Accuracy: {self.best_test_metrics['accuracy']:.3f}%")
            self.logger.info(f"  Best AUC: {self.best_test_metrics['auc']:.4f}")
            self.logger.info(f"  Best Sensitivity: {self.best_test_metrics['sensitivity']:.4f}")
            self.logger.info(f"  Best Specificity: {self.best_test_metrics['specificity']:.4f}")
            self.logger.info(f"  Best Precision: {self.best_test_metrics['precision']:.4f}")
            self.logger.info(f"  Best Recall: {self.best_test_metrics['recall']:.4f}")
            self.logger.info(f"  Best F1: {self.best_test_metrics['f1']:.4f}")
            self.logger.info(f"{'='*60}")
        else:
            self.logger.info(f"\n{'='*60}")
            self.logger.info("Training Final Results:")
            self.logger.info(f"  Best Epoch: {self.fold_results['best_epoch']}")
            self.logger.info(f"  Best Accuracy: {self.best_test_metrics['accuracy']:.3f}%")
            self.logger.info(f"  Best AUC: {self.best_test_metrics['auc']:.4f}")
            self.logger.info(f"  Best Sensitivity: {self.best_test_metrics['sensitivity']:.4f}")
            self.logger.info(f"  Best Specificity: {self.best_test_metrics['specificity']:.4f}")
            self.logger.info(f"  Best Precision: {self.best_test_metrics['precision']:.4f}")
            self.logger.info(f"  Best Recall: {self.best_test_metrics['recall']:.4f}")
            self.logger.info(f"  Best F1: {self.best_test_metrics['f1']:.4f}")
            self.logger.info(f"{'='*60}")
        
        return self.fold_results