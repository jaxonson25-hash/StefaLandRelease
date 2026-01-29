from typing import Optional
import numpy as np
import torch
from numpy.typing import NDArray
from dmg.core.data.data import random_index
from dmg.core.data.samplers.base import BaseSampler

class HydroSampler(BaseSampler):
    """Simplified sampler with fixed multi-day forecasting - no curriculum learning."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.device = config['device']
        self.warm_up = config['delta_model']['phy_model']['warm_up']
        self.rho = config['delta_model']['rho']
        
        # Fixed forecast days - no curriculum
        self.forecast_days = config.get('forecast_days', 1)  # Default to 1 for backwards compatibility
        
        # For fire models, check if we should use multi-day forecasting
        if self._is_fire_model(config):
            self.forecast_days = config.get('fire_forecast_days', 10)  # Default 10 days for fire
        
    def _is_fire_model(self, config: dict) -> bool:
        """Check if this is a fire prediction model."""
        # Check various indicators that this is fire prediction
        target = config.get('train', {}).get('target', [])
        if any('fire' in str(t).lower() for t in target):
            return True
        
        nn_model = config.get('delta_model', {}).get('nn_model', {}).get('model', '')
        if 'fire' in nn_model.lower():
            return True
            
        return False

    def load_data(self):
        """Custom implementation for loading data."""
        raise NotImplementedError

    def preprocess_data(self):
        """Custom implementation for preprocessing data."""
        raise NotImplementedError
    
    def select_subset(
        self,
        x: torch.Tensor,
        i_grid: NDArray[np.float32],
        i_t: Optional[NDArray[np.float32]] = None,
        c: Optional[NDArray[np.float32]] = None,
        tuple_out: bool = False,
        has_grad: bool = False,
        forecast_days: Optional[int] = None,  # Allow override
    ) -> torch.Tensor:
        """Select subset with multi-day support - backwards compatible."""
        
        # Use instance forecast_days if not overridden
        if forecast_days is None:
            forecast_days = self.forecast_days
            
        batch_size, nx = len(i_grid), x.shape[-1]
        
        # Determine if this is fire/grid data or hydro/basin data
        is_fire_data = self._is_fire_model(self.config)
        
        if i_t is not None:
            # Standard sequence length for input
            input_timesteps = self.rho + self.warm_up
            
            x_tensor = torch.zeros(
                [input_timesteps, batch_size, nx],
                device=self.device,
                requires_grad=has_grad,
            )
            
            for k in range(batch_size):
                idx = int(i_grid[k])
                start_idx = int(i_t[k]) - self.warm_up
                end_idx = start_idx + input_timesteps
                
                # Bounds checking
                if idx >= x.shape[1]:
                    print(f"Warning: Index {idx} exceeds data dimension {x.shape[1]}")
                    continue
                
                # Ensure we don't exceed time bounds
                if end_idx <= x.shape[0] and start_idx >= 0:
                    if is_fire_data:
                        # Fire data: use slice indexing for grid compatibility
                        x_tensor[:, k:k + 1, :] = x[start_idx:end_idx, idx:idx + 1, :]
                    else:
                        # Hydro data: use direct indexing for basins
                        if x.ndim == 3:
                            x_tensor[:, k, :] = x[start_idx:end_idx, idx, :]
                        else:
                            # Handle 2D case: [time, features] - single basin
                            x_tensor[:, k, :] = x[start_idx:end_idx, :]
                else:
                    # Handle edge case - use available data and pad
                    if start_idx >= 0 and start_idx < x.shape[0]:
                        if is_fire_data:
                            available_data = x[start_idx:, idx:idx + 1, :]
                            if available_data.shape[0] > 0 and available_data.shape[1] > 0:
                                avail_len = min(available_data.shape[0], input_timesteps)
                                x_tensor[:avail_len, k:k + 1, :] = available_data[:avail_len]
                                # Pad with last timestep if needed
                                if avail_len < input_timesteps:
                                    x_tensor[avail_len:, k:k + 1, :] = available_data[-1:].repeat(
                                        input_timesteps - avail_len, 1, 1
                                    )
                        else:
                            # Hydro data padding
                            if x.ndim == 3:
                                available_data = x[start_idx:, idx, :]
                            else:
                                available_data = x[start_idx:, :]
                                
                            if available_data.shape[0] > 0:
                                avail_len = min(available_data.shape[0], input_timesteps)
                                x_tensor[:avail_len, k, :] = available_data[:avail_len]
                                # Pad with last timestep if needed
                                if avail_len < input_timesteps:
                                    if x.ndim == 3:
                                        x_tensor[avail_len:, k, :] = available_data[-1:].repeat(
                                            input_timesteps - avail_len, 1
                                        )
                                    else:
                                        x_tensor[avail_len:, k, :] = available_data[-1].repeat(
                                            input_timesteps - avail_len, 1
                                        )
        else:
            # Static indexing without time dimension
            if x.ndim == 3:
                x_tensor = x[:, i_grid, :].float().to(self.device)
            else:
                x_tensor = x[i_grid, :].float().to(self.device)
        
        if c is not None:
            c_tensor = torch.from_numpy(c).float().to(self.device)
            repeat_timesteps = input_timesteps if i_t is not None else self.rho + self.warm_up
            c_tensor = c_tensor[i_grid].unsqueeze(1).repeat(1, repeat_timesteps, 1)
            return (x_tensor, c_tensor) if tuple_out else torch.cat((x_tensor, c_tensor), dim=2)
        
        return x_tensor
    
    def create_multi_day_targets(
        self, 
        targets: torch.Tensor, 
        i_t: NDArray[np.float32],
        i_grid: NDArray[np.float32]
    ) -> torch.Tensor:
        """Create multi-day targets for forecasting."""
        batch_size = len(i_grid)
        
        if self.forecast_days == 1:
            # Backwards compatible - return standard targets
            return self.select_subset(targets, i_grid, i_t)[self.warm_up:, :]
        
        # Multi-day targets: [batch_size, forecast_days]
        multi_targets = torch.zeros(
            [batch_size, self.forecast_days],
            device=self.device
        )
        
        # Determine if this is fire/grid data or hydro/basin data
        is_fire_data = self._is_fire_model(self.config)
        
        for k in range(batch_size):
            idx = int(i_grid[k])
            if idx >= targets.shape[1]:
                print(f"Warning: Index {idx} exceeds target dimension {targets.shape[1]}")
                continue
                
            for day in range(self.forecast_days):
                # Target starts right after the input sequence
                target_idx = int(i_t[k]) + day
                if target_idx < targets.shape[0]:
                    if targets.ndim == 3:
                        multi_targets[k, day] = targets[target_idx, idx, 0]
                    else:
                        # For 2D targets: [time, basin/grid]
                        multi_targets[k, day] = targets[target_idx, idx]
                else:
                    # If we exceed available data, use last available value
                    if targets.ndim == 3:
                        multi_targets[k, day] = targets[-1, idx, 0]
                    else:
                        multi_targets[k, day] = targets[-1, idx]
        
        return multi_targets
    
    def get_training_sample(
        self,
        dataset: dict[str, NDArray[np.float32]],
        ngrid_train: int,
        nt: int,
    ) -> dict[str, torch.Tensor]:
        """Generate training batch with fixed multi-day forecasting."""
        batch_size = self.config['train']['batch_size']
        
        # Adjust max time to ensure we have enough future data for targets
        max_start_time = nt - self.forecast_days
        
        # Add debugging to check data dimensions
        # print(f"Debug: Dataset target shape: {dataset['target'].shape}")
        # print(f"Debug: ngrid_train: {ngrid_train}, nt: {nt}")
        # print(f"Debug: batch_size: {batch_size}")
        
        # Sample random indices
        i_sample, i_t = random_index(
            ngrid_train, 
            max_start_time, 
            (batch_size, self.rho), 
            warm_up=self.warm_up
        )
        
        # print(f"Debug: i_sample range: {i_sample.min()} to {i_sample.max()}")
        # print(f"Debug: i_t range: {i_t.min()} to {i_t.max()}")
        
        # Create targets based on forecast days
        if self.forecast_days > 1:
            targets = self.create_multi_day_targets(dataset['target'], i_t, i_sample)
        else:
            # Single day - backwards compatible
            targets = self.select_subset(dataset['target'], i_sample, i_t)[self.warm_up:, :]
        
        # Standard inputs (same length regardless of forecast days)
        xc_nn_norm = self.select_subset(
            dataset['xc_nn_norm'], 
            i_sample, 
            i_t, 
            has_grad=False
        )[self.warm_up:self.warm_up + self.rho, :]
        
        # Build batch data dictionary
        batch_data = {
            'x_phy': self.select_subset(dataset['x_phy'], i_sample, i_t),
            'c_phy': dataset['c_phy'][i_sample],
            'c_nn': dataset['c_nn'][i_sample],
            'xc_nn_norm': xc_nn_norm,
            'target': targets,
            'batch_sample': i_sample,
            'current_forecast_days': self.forecast_days,  # Always fixed now
        }
        
        return batch_data
    
    def get_validation_sample(self, dataset, i_s, i_e):
        rho = self.rho
        warm_up = self.warm_up

        # Evaluation uses fixed rolling windows
        i_grid = np.arange(i_s, i_e)
        i_t = np.full(len(i_grid), rho + warm_up)

        sample = {
            'xc_nn_norm': self.select_subset(
                dataset['xc_nn_norm'], i_grid, i_t
            )[warm_up:warm_up + rho],

            'target': self.select_subset(
                dataset['target'], i_grid, i_t
            )[warm_up:warm_up + rho],

            'c_nn': dataset['c_nn'][i_grid],
            'c_phy': dataset['c_phy'][i_grid],
            'temporal_features': dataset['temporal_features'],
        }

        return {
            k: v.to(dtype=torch.float32, device=self.device)
            for k, v in sample.items()
        }

    
    def get_current_forecast_horizon(self):
        """Get current forecast horizon - now always fixed."""
        return self.forecast_days
    
    def is_curriculum_enabled(self):
        """Check if curriculum learning is enabled - always False now."""
        return False