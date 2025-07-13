from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_loaders(data_dir, img_size=224, batch_size=32):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.4591, 0.4753, 0.4116],[0.1812, 0.1573, 0.1957] ),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=tf)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

# Computed mean: tensor([0.4591, 0.4753, 0.4116])
# Computed std:  tensor([0.1812, 0.1573, 0.1957])