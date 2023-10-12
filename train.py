import argparse
import logging
import wandb
from glob import glob
from tqdm import tqdm
import pandas as pd
import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.data_loading import MriDataset
from utils.utils import BCE_dice, dice_pytorch, get_file_row, iou_pytorch, EarlyStopping
from test import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 1e-3
):
    # Read csv file of data
    files_dir = 'kaggle_3m'
    file_paths = glob(f'{files_dir}/*/*[0-9].tif')

    csv_path = 'kaggle_3m/data.csv'
    df = pd.read_csv(csv_path)

    # Missing values handling
    imputer = SimpleImputer(strategy="most_frequent")

    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    filenames_df = pd.DataFrame((get_file_row(filename) for filename in file_paths), columns=['Patient', 'image_filename', 'mask_filename'])

    df = pd.merge(df, filenames_df, on="Patient")

    # Split data into Train, valid, test
    train_df, test_df = train_test_split(df, test_size=0.3)
    test_df, valid_df = train_test_split(test_df, test_size=0.5)

    # Transforming 
    transform = A.Compose([
        A.ChannelDropout(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
    ])

    train_dataset = MriDataset(train_df, transform)
    valid_dataset = MriDataset(valid_df)
    test_dataset = MriDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # (Initialize logging)
    experiment = wandb.init(project='EfficientUNet', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    )
    global_step = 0

    # history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    early_stopping = EarlyStopping(patience=7)
    loss_fn = BCE_dice
    optimizer = Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=2,factor=0.2)

    for epoch in range(1, epochs + 1): 
        running_loss = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            loss = loss_fn(predictions, mask)
            running_loss += loss.item() * img.size(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            experiment.log({
                'train/learning rate': optimizer.param_groups[0]['lr'],
                'train/train_loss': loss,
                # 'images': wandb.Image(images[0].cpu()),
                # 'masks': {
                #     'true': wandb.Image(true_masks[0].float().cpu()),
                #     'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                # },
                'train/step': global_step,
                'train/epoch': epoch
            })
        
        model.eval()
        with torch.no_grad():
            running_IoU = 0
            running_dice = 0
            running_valid_loss = 0
            for i, data in enumerate(valid_loader):
                img, mask = data
                img, mask = img.to(device), mask.to(device)
                predictions = model(img)
                predictions = predictions.squeeze(1)
                running_dice += dice_pytorch(predictions, mask).sum().item()
                running_IoU += iou_pytorch(predictions, mask).sum().item()
                loss = loss_fn(predictions, mask)
                loss = loss.item() * img.size(0)
                running_valid_loss += loss

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = running_valid_loss / len(valid_loader.dataset)
        val_dice = running_dice / len(valid_loader.dataset)
        val_IoU = running_IoU / len(valid_loader.dataset)
        experiment.log({
            'eval/validation Dice': val_dice,
            'eval/validation IOU': val_IoU,
            'eval/val_loss': val_loss
        })
        print(f'Epoch: {epoch}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss} | Validation Mean IoU: {val_IoU} '
            f'| Validation Dice coefficient: {val_dice}')
        
        lr_scheduler.step(val_loss)
        if early_stopping(val_loss, model):
            early_stopping.load_weights(model)
            break
    model.eval()
    test(model, test_loader, device)
    experiment.finish()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    # parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
    #                     help='Percent of the data that is used as validation (0-100)')
    # parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    # parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=1 for grey scale images
    # n_classes is the number of probabilities you want to get per pixel
    # load model
    model = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid',
    )
    model.to(device)

    logging.info(f'Network:\n'
                 f'\t{3} input channels\n'
                 f'\t{1} output channels (classes)\n')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
        )
