import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings
from torchvision import datasets, transforms
from omegaconf import OmegaConf
from equiadapt.images.canonicalization.discrete_group import GroupEquivariantImageCanonicalization
from equiadapt.images.canonicalization_networks import ESCNNEquivariantNetwork

# Suppress e2cnn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="e2cnn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize the C8 Steerable Canonicalization Network
# The FOCAL paper benchmarks use C8 (8 rotations) for ResNet50 
canonicalization_network = ESCNNEquivariantNetwork(
    (3, 224, 224),  # in_shape
    16,             # out_channels (width)
    3,              # kernel_size
    "rotation",     # group_type
    8,              # num_rotations (C8 group for 45-degree intervals) [cite: 1199]
    3               # num_layers
).to(device)

# 2. Define Hyperparameters according to the PRLC Paper [cite: 1187, 1400]
canonicalization_hyperparams = OmegaConf.create({
    "group_type": "rotation",
    "num_rotations": 8,
    "alpha": 1.0, 
    "beta": 100.0,              # Prior weight beta=100 as per PRLC paper [cite: 1187]
    "input_crop_ratio": 0.9,    # Prevents boundary artifacts during rotation
    "resize_shape": [224, 224]
})

# 3. Initialize the PRLC Aligner
canonicalizer = GroupEquivariantImageCanonicalization(
    canonicalization_network=canonicalization_network,
    canonicalization_hyperparams=canonicalization_hyperparams,
    in_shape=(3, 224, 224)
).to(device)

# 4. Load the ResNet50 backbone (Pre-trained on ImageNet-1K) [cite: 1181, 1182]
prediction_network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)

# 5. Joint Optimizer for Equivariant Adaptation 
optimizer = optim.Adam(
    list(prediction_network.parameters()) + list(canonicalization_network.parameters()), 
    lr=1e-4  # Learning rate typically used for fine-tuning
)

# 6. Dataset setup (CIFAR10 used in the FOCAL/PRLC benchmarks) [cite: 355, 1182]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root="./datasets", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 7. Training Loop: Joint Training Protocol [cite: 1185, 1186]
prediction_network.train()
canonicalization_network.train()

for epoch in range(100): # PRLC is typically trained until convergence [cite: 1084]
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Canonicalize the input
        inputs_canonicalized = canonicalizer(inputs) 
        
        # Pass to the ResNet50 backbone
        outputs = prediction_network(inputs_canonicalized) 

        # PRLC Total Loss: Task Cross-Entropy + Beta * Prior Loss [cite: 1186, 1187]
        loss = F.cross_entropy(outputs, targets)
        loss += canonicalization_hyperparams.beta * canonicalizer.get_prior_regularization_loss() 

        loss.backward()
        optimizer.step()