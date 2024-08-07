import torch
import os
import argparse
import data_loader, data_setup, model_builder, engine
from dotenv import load_dotenv
from torchvision import transforms
from vision_transformer import FullNetwork, ModelArgs

load_dotenv()

# Setting up script time parameters.
parser = argparse.ArgumentParser(description="This script is responsible for training the model!!")
arguments = {
    '--model_name': {'type': str, 'help': 'please specify a model name for saving', 'default': 'vision.pt'},
    '--use_pretrained': {'type': bool, 'help': 'Hit True if you want to use a pretrained model', 'default': False}
}
for arg, kwargs in arguments.items():
    parser.add_argument(arg, **kwargs)

args = parser.parse_args()

# Setting up the hyper params
HIDDEN_UNITS = 64 # Param for units in hidden layers
BATCH_SIZE = 8
EVAL_INTERVAL = 8 # Initially setting it will change accordingly
EVAL_ITERS = 8
NUM_EPOCHS = 200
VALIDATION_SPLIT = 0.3
CHANNELS = 3
LEARNING_RATE = 3e-4

# Setup directories
train_dir = os.path.join(os.environ.get('data_path', '/home/syednoor/Desktop/FAIR/Vision-Transformer/data'), 'images', 'train')
test_dir = os.path.join(os.environ.get('data_path','/home/syednoor/Desktop/FAIR/Vision-Transformer/data'), 'images', 'test')

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, val_dataloader, test_dataloader, classes = data_loader.build_dataloader(
    train_dir=train_dir, test_dir=test_dir, batch_size=BATCH_SIZE, transform=data_transform ,validation_split=VALIDATION_SPLIT
)

margs = ModelArgs()
# Let's try to create our model, optimizer and loss function
model = FullNetwork(margs)
model = model.to(device=device)

# Cross entropy loss as we're dealing with multiclass
loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters() ,lr=LEARNING_RATE ,weight_decay=1e-2)

# Training the model 

engine.train(model=model,
             train_dataloader=train_dataloader,
             validation_dataloader=val_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_function=loss_function,
             epochs=NUM_EPOCHS,
             eval_iters=EVAL_ITERS,
             eval_interval=EVAL_INTERVAL,
             model_name=args.model_name,
             device=device)