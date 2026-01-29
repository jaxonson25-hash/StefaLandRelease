class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, **kwargs):
        mask = ~torch.isnan(y)
        y_hat_masked = y_hat[mask]
        y_masked = y[mask]
        
        # Apply sigmoid
        p = torch.sigmoid(y_hat_masked)
        
        # Focal loss calculation
        ce_loss = F.binary_cross_entropy(p, y_masked, reduction='none')
        p_t = p * y_masked + (1 - p) * (1 - y_masked)
        alpha_t = self.alpha * y_masked + (1 - self.alpha) * (1 - y_masked)
        
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()