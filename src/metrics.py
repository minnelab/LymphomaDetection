import torch
from torch.nn import functional as F
from monai.losses import ContrastiveLoss


class TotalLoss:
    def __init__(self, recon_loss, contrastive_loss):
        self.recon_loss = recon_loss
        self.contrastive_loss = contrastive_loss

    def __call__(self, outputs, target):
        r_loss = self.recon_loss(outputs["outputs_v1"], target)
        cl_loss = self.contrastive_loss(outputs["flat_out_v1"], outputs["flat_out_v2"])

        total_loss = r_loss + cl_loss * r_loss

        return {"total_loss": total_loss, "recon_loss": r_loss, "contrastive_loss": cl_loss}
    

def recon_val_loss_transform(output):
   outputs_v1 = [out["pred"] for out in output]  
   gt_inputs = [out["label"] for out in output]
   
   return  outputs_v1, gt_inputs

def recon_loss_transform(output):
   outputs_v1 = [out["pred"]["outputs_v1"] for out in output]  
   gt_inputs = [out["label"] for out in output]
   
   return  outputs_v1, gt_inputs

def contrastive_loss_transform(output):
   flat_out_v1 = [out["pred"]["flat_out_v1"] for out in output]
   flat_out_v2 = [out["pred"]["flat_out_v2"] for out in output]
   
   flat_out_v1 = torch.stack(flat_out_v1, dim=0)
   flat_out_v2 = torch.stack(flat_out_v2, dim=0)
   return  flat_out_v1, flat_out_v2


class AMPContrastiveLoss(ContrastiveLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast():
            return super().forward(input, target)