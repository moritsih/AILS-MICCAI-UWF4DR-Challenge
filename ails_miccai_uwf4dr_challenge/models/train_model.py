import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from models.task1_automorph import AutoMorphModel
from torch.utils.data import DataLoader
from ails_miccai_uwf4dr_challenge.augmentations import rotate_affine_flip_choice, resize_only
from ails_miccai_uwf4dr_challenge.dataset import DatasetBuilder, CustomDataset
import typer

app = typer.Typer()

@app.command()
def main():
    # Set PYTORCH_CUDA_ALLOC_CONF
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    progress_bar = TQDMProgressBar(refresh_rate=10)

    dataset_builder = DatasetBuilder(dataset='all', task='task1')
    train_data, val_data = dataset_builder.get_train_val()

    train_dataset = CustomDataset(train_data, transform=rotate_affine_flip_choice)
    val_dataset = CustomDataset(val_data, transform=resize_only)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, persistent_workers=True)

    # Initialize the model
    model = AutoMorphModel()

    # Updated Trainer initialization
    trainer = L.Trainer(
        accelerator='gpu',
        devices=1,  # number of GPUs
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar],
        log_every_n_steps=1,  # Adjust the logging interval
        precision=16,  # Enable mixed precision training
    )

    # Fit the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    app()
