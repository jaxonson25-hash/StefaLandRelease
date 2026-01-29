import logging
import torch
from typing import Any, Dict, Optional
import numpy as np
from numpy.typing import NDArray
from dmg.core.data.data import random_index
from dmg.core.data.samplers.base import BaseSampler

log = logging.getLogger(__name__)


class CurriculumManager:
    """Curriculum learning manager with proper multi-day target creation."""
    
    def __init__(self, config: Dict[str, Any]):
        curriculum_config = config.get('curriculum', {})
        
        self.enabled = curriculum_config.get('enabled', False)
        self.max_forecast_days = curriculum_config.get('max_forecast_days', 10)
        self.start_days = curriculum_config.get('start_days', 1)
        self.strategy = curriculum_config.get('strategy', 'step')
        self.step_epochs = curriculum_config.get('step_epochs', 10)
        self.warmup_epochs = curriculum_config.get('warmup_epochs', 5)
        
        self.current_forecast_days = self.start_days
        
        if self.enabled:
            log.info(f"Curriculum learning enabled: {self.start_days} -> {self.max_forecast_days} days")
    
    def update_curriculum(self, epoch: int, total_epochs: int) -> bool:
        """Update curriculum based on epoch."""
        if not self.enabled or epoch <= self.warmup_epochs:
            return False
            
        old_days = self.current_forecast_days
        
        if self.strategy == 'step':
            effective_epoch = epoch - self.warmup_epochs
            if effective_epoch > 0 and effective_epoch % self.step_epochs == 0:
                self.current_forecast_days = min(
                    self.max_forecast_days,
                    self.current_forecast_days + 1
                )
        elif self.strategy == 'linear':
            effective_epoch = epoch - self.warmup_epochs
            effective_total = total_epochs - self.warmup_epochs
            if effective_total > 0:
                progress = effective_epoch / effective_total
                self.current_forecast_days = min(
                    self.max_forecast_days,
                    int(self.start_days + progress * (self.max_forecast_days - self.start_days))
                )
        
        if old_days != self.current_forecast_days:
            log.info(f"Curriculum updated: {old_days} -> {self.current_forecast_days} days")
            return True
        return False
    
    def get_current_horizon(self) -> int:
        """Get current forecast horizon."""
        return self.current_forecast_days


class MultiStepLossManager:
    """Multi-step loss manager with proper day weighting."""
    
    def __init__(self, config: Dict[str, Any]):
        curriculum_config = config.get('curriculum', {})
        max_days = curriculum_config.get('max_forecast_days', 10)
        decay_rate = curriculum_config.get('weight_decay_rate', 0.9)
        
        # Create day weights with exponential decay
        weights = [decay_rate ** i for i in range(max_days)]
        self.day_weights = torch.tensor(weights, dtype=torch.float32)
        self.day_weights = self.day_weights / self.day_weights.sum()
    
    def compute_multi_step_loss(self, predictions, targets, loss_func, current_horizon: int):
        """Compute weighted loss for multi-step predictions."""
        if current_horizon == 1:
            # Single step - standard loss (backward compatible)
            return loss_func(predictions, targets)
        
        # Debug shapes
        # print(f"Loss computation - predictions: {predictions.shape}, targets: {targets.shape}, horizon: {current_horizon}")
        
        # Multi-step loss computation
        if predictions.dim() == 2 and predictions.shape[1] == current_horizon:
            # predictions: [batch_size, forecast_days]
            # targets: [batch_size, forecast_days] 
            if targets.dim() == 2 and targets.shape[1] == current_horizon:
                total_loss = 0.0
                active_weights = self.day_weights[:current_horizon].to(predictions.device)
                
                for day in range(current_horizon):
                    day_pred = predictions[:, day:day+1]  # [batch_size, 1]
                    day_target = targets[:, day:day+1]    # [batch_size, 1]
                    
                    day_loss = loss_func(day_pred, day_target)
                    total_loss += active_weights[day] * day_loss
                
                return total_loss
            else:
                # Targets wrong shape - use first day only
                day_pred = predictions[:, 0:1]  # [batch_size, 1]  
                if targets.dim() == 3:
                    day_target = targets[:, :, 0:1] if targets.shape[2] == 1 else targets[:, 0:1, :]
                elif targets.dim() == 2:
                    day_target = targets[:, 0:1] if targets.shape[1] > 1 else targets
                else:
                    day_target = targets.unsqueeze(-1)
                
                return loss_func(day_pred, day_target)
        else:
            # Fallback to standard loss
            return loss_func(predictions, targets)