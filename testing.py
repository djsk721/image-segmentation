import torch
import wandb
from scipy.ndimage import binary_dilation


@torch.inference_mode()
def test(model, dataloader, device, experiment):
    tbl = wandb.Table(columns=["image", "prediction", "label"])
    with torch.no_grad():
        for data in dataloader:
            image, mask = data
            mask = mask[0]
            if not mask.byte().any():
                continue
            image = image.to(device)
            gt_image = image
            pred_img = image
            prediction = model(image).to('cpu')[0][0]
            prediction = torch.where(prediction > 0.5, 1, 0)
            prediction_edges = prediction - binary_dilation(prediction)
            ground_truth = mask - binary_dilation(mask)
            gt_image[0, 0, ground_truth.bool()] = 1
            pred_img[0, 1, prediction_edges.bool()] = 1
            tbl.add_data(wandb.Image(image.cpu()), wandb.Image(pred_img.cpu()),
                          wandb.Image(gt_image.cpu()))
        experiment.log({"table": tbl})