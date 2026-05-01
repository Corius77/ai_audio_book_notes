import torch
import myk_models

saved_pth_path = "lstm_size_32_epoch_413_loss_0.0013.pth"
export_pt_path = "dist_32.pt"

model = torch.load(saved_pth_path, weights_only=False, map_location='cpu')
model.eval()

scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, export_pt_path)
