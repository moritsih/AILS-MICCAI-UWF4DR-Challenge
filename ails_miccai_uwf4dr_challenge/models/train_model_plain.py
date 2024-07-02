import torch
import torch.nn as nn
import torch.optim as optim


from ails_miccai_uwf4dr_challenge.dataset import CustomDataset, DatasetBuilder
from ails_miccai_uwf4dr_challenge.augmentations import rotate_affine_flip_choice, resize_only
from torch.utils.data import DataLoader

from ails_miccai_uwf4dr_challenge.models.trainer import Trainer
from ails_miccai_uwf4dr_challenge.models.architectures.task1_automorph_plain import AutoMorphModel

def main():
    dataset = DatasetBuilder(dataset='all', task='task1')
    train_data, val_data = dataset.get_train_val()

    train_dataset = CustomDataset(train_data, transform=rotate_affine_flip_choice)
    val_dataset = CustomDataset(val_data, transform=resize_only)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = AutoMorphModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

    num_epochs = 10
    trainer.train(num_epochs)

if __name__ == "__main__":
    main()
