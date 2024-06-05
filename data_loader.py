import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

NUM_WORKERS = os.cpu_count()

def build_dataloader(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        validation_split: float = 0.2,
        num_workers: int = NUM_WORKERS
):
  
  # This loads the data from directories automatically taking the folder names as classnames.
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # This ensures on fixing a bound for validation_split
  assert validation_split < 1, "The validation split should be lesser than 1 !!!. 0.8 implies 80% data pushed to validation rest for train"
  
  # Introducing validation set from train samples as do not want to disturb the test set.
  val_size = int(validation_split * len(train_data))
  train_size = int(len(train_data) - val_size)

  # This function randomly samples train_size number of samples to train_sub and viceversa for val_sub.
  train_sub, val_sub = random_split(train_data, [train_size, val_size])

  # Get Image classes
  image_classes = train_data.classes

  # Produce image generators
  train_dataloader = DataLoader(
    train_sub,
    batch_size=batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
  )

  validation_dataloader = DataLoader(
    val_sub,
    batch_size=batch_size,
    shuffle=False, # Not shuffling validation and test as not required!!
    num_workers=NUM_WORKERS,
    pin_memory=True
  )

  test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
  )

  return train_dataloader, validation_dataloader, test_dataloader, image_classes
