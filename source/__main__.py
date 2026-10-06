import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import numpy as np
from typing import List, Dict

from .dataset import dataset_factory
from .models import model_factory
from .components import optimizers_factory, lr_scheduler_factory
from .training import Train

import os
import torch
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def model_training(cfg: DictConfig):
    """单折训练版本"""
    dataloaders = dataset_factory(cfg)
    model = model_factory(cfg)
    print(model)
    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    
    # 创建训练实例
    training = Train(cfg=cfg,
                     model=model,
                     optimizers=optimizers,
                     lr_schedulers=lr_schedulers,
                     dataloaders=dataloaders)
    training.train()

def model_training_10fold(cfg: DictConfig):       
    """十折交叉验证版本"""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) 
    
    # 获取十折数据
    fold_dataloaders = dataset_factory(cfg)
    
    # 确保有10折数据
    assert len(fold_dataloaders) == 10, f"Expected 10 folds, got {len(fold_dataloaders)}"
    
    all_fold_results = []
    logger = logging.getLogger()
    
    for fold, dataloaders in enumerate(fold_dataloaders):
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Fold {fold + 1}/10")
        logger.info(f"{'='*80}")
        
        # 创建新的模型实例
        model = model_factory(cfg)
        optimizers = optimizers_factory(
            model=model, optimizer_configs=cfg.optimizer)
        lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
        
        # 创建训练实例
        training = Train(cfg=cfg,
                         model=model,
                         optimizers=optimizers,
                         lr_schedulers=lr_schedulers,
                         dataloaders=dataloaders)
        
        # 设置当前折信息
        training.set_fold_info(fold_idx=fold, total_folds=10)
        
        # 训练并获取结果
        fold_result = training.train()
        all_fold_results.append(fold_result)
    
    # 所有折训练完成后，在日志中输出汇总结果
    log_cross_validation_summary(all_fold_results, logger)
    
    return all_fold_results

def log_cross_validation_summary(all_fold_results: List[Dict], logger):
    """在日志中输出十折交叉验证的汇总结果"""
    logger.info(f"\n{'='}")
    logger.info("10-FOLD CROSS VALIDATION FINAL RESULTS SUMMARY")
    logger.info(f"{'='}")
    
    # 收集所有折的最佳指标
    metrics_to_collect = [
        "accuracy", "auc", "sensitivity", "specificity", 
        "precision", "recall", "f1"
    ]
    
    # 存储每个指标的所有折结果
    all_metrics_values = {metric: [] for metric in metrics_to_collect}
    
    # 打印表头 - 调整宽度以适应3位小数
    header = "Fold | " + " | ".join([f"{m:>9}" for m in ["Acc(%)", "AUC", "Sen", "Spe", "Prec", "Rec", "F1"]])
    logger.info(header)
    logger.info("-" * len(header))
    
    # 打印每个折的结果
    for i, fold_result in enumerate(all_fold_results):
        if fold_result.get("best_metrics"):
            best = fold_result["best_metrics"]
            fold_str = f"{i+1:4d} | "
            
            for metric in metrics_to_collect:
                if metric == "accuracy":
                    value = best.get("test_accuracy", 0)
                else:
                    value = best.get(metric, 0)
                
                if metric == "accuracy":
                    value_str = f"{value:8.3f}%"  # 8.3f显示3位小数
                else:
                    value_str = f"{value:8.4f}"      # 8.4f显示4位小数
                
                fold_str += f"{value_str} | "
                all_metrics_values[metric].append(value)
            
            logger.info(fold_str)
    
    logger.info("-" * len(header))
    
    # 计算并打印统计信息
    logger.info("\nOVERALL STATISTICS (10-Fold Cross Validation):")
    logger.info("-" * 80)
    
    for metric in metrics_to_collect:
        values = all_metrics_values[metric]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            median_val = np.median(values)
            
            if metric == "accuracy":
                # 准确率显示为百分比，3位小数
                logger.info(f"{metric.upper():12s}: {mean_val:7.3f}% ± {std_val:7.3f}%  "
                          f"(Min: {min_val:7.3f}%, Max: {max_val:7.3f}%, Median: {median_val:7.3f}%)")
            else:
                logger.info(f"{metric.upper():12s}: {mean_val:7.4f} ± {std_val:7.4f}  "
                          f"(Min: {min_val:7.4f}, Max: {max_val:7.4f}, Median: {median_val:7.4f})")
    
    # 打印最佳折的信息
    logger.info("\nBEST PERFORMING FOLDS:")
    logger.info("-" * 80)
    
    for metric in metrics_to_collect:
        if all_metrics_values[metric]:
            best_fold_idx = np.argmax(all_metrics_values[metric])
            best_value = all_metrics_values[metric][best_fold_idx]
            
            if metric == "accuracy":
                logger.info(f"{metric.upper():12s}: Fold {best_fold_idx+1} with {best_value:7.3f}%")
            else:
                logger.info(f"{metric.upper():12s}: Fold {best_fold_idx+1} with {best_value:7.4f}")
    
    logger.info(f"\n{'='}")
    logger.info("END OF 10-FOLD CROSS VALIDATION")
    logger.info(f"{'='}")

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 设置设备
    if cfg.device == 'cuda' and torch.cuda.is_available():
        cfg.device = 'cuda'
    else:
        cfg.device = 'cuda:1'
    
    # 创建wandb组名
    data_creation = cfg.model.get('data_creation', {})
    if data_creation == "hypergraph":
        group_name = f"{cfg.dataset.name}_{cfg.model.name}_node:{cfg.dataset.node}_Kneigs:{cfg.model.K_neigs}_layers:{cfg.model.num_layers}_hidden:{cfg.model.hidden_size}_lr:exp_1e-5"
    else:
        group_name = f"{cfg.dataset.name}_{cfg.model.name}_node:{cfg.dataset.node}_layers:{cfg.model.num_layers}_hidden:{cfg.model.hidden_size}_lr:exp_1e-5"

    # 重复实验次数
    for repeat in range(cfg.get("repeat_time", 1)):
        logger = logging.getLogger()
        logger.info(f"\n{'#'*80}")
        logger.info(f"Experiment Repeat {repeat + 1}/{cfg.get('repeat_time', 1)}")
        logger.info(f"{'#'*80}")
        
        if cfg.is_wandb:
            run = wandb.init(
                project=cfg.project, 
                reinit=True,
                group=f"{group_name}", 
                tags=[f"{cfg.dataset.name}", f"{cfg.model.name}"],
                config=OmegaConf.to_container(cfg, resolve=True)
            )
        
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        
        # 检查是否需要十折交叉验证
        if cfg.get("use_10fold_cv", True):
            logger.info("Using 10-Fold Cross Validation")
            all_fold_results = model_training_10fold(cfg)
        else:
            logger.info("Using Single Fold Training")
            model_training(cfg)

        if cfg.is_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()