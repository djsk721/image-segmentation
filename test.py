import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import binary_dilation


@torch.inference_mode()
def test(model, dataloader, device, experiment):
    with torch.no_grad():
        for data in dataloader:
            image, mask = data
            mask = mask[0]
            if not mask.byte().any():
                continue
            image = image.to(device)
            prediction = model(image).to('cpu')[0][0]
            prediction = torch.where(prediction > 0.5, 1, 0)
            prediction_edges = prediction - binary_dilation(prediction)
            ground_truth = mask - binary_dilation(mask)
            image[0, 0, ground_truth.bool()] = 1
            image[0, 1, prediction_edges.bool()] = 1
            return image