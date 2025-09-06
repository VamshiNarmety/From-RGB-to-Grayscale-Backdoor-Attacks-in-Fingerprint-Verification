import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
class TriggerGenerator(nn.Module):
    def __init__(self):
        super(TriggerGenerator, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
    
    def forward(self, z):
        z = self.conv1(z)
        z = F.relu(z)
        z = self.conv2(z)
        z = F.relu(z)
        z = self.conv3(z)
        return z
    
def create_kernel(v):
    size = 2*v+1
    kernel = torch.full((size, size), -1.0)
    kernel[v, v] = ((size)*(size))-1
    kernel = kernel.expand(1, 1, size, size)
    return kernel


def trigger_loss(trigger, kernel):
    kernel = kernel.to(trigger.device)
    conv_res =  F.conv2d(trigger, kernel, stride=1, padding=1, groups=1)
    norm = torch.norm(conv_res, p=1)
    return -torch.log(norm)


def apply_pff_trigger_tensor(image_tensor, generator_path, a=0.1, z_seed=None, clamp_val=0.1, radius=70):
    assert image_tensor.dim() == 4 and image_tensor.shape[1] == 1, "Expected grayscale image tensor of shape (N, 1, H, W)"
    device = image_tensor.device
    model = TriggerGenerator().to(device)
    model.load_state_dict(torch.load(generator_path, map_location=device))
    model.eval()
    if z_seed is not None:
        torch.manual_seed(z_seed)
    z = torch.randn_like(image_tensor)
    with torch.no_grad():
        delta = model(z)
        delta = torch.clamp(delta, -clamp_val, clamp_val)
    N, C, H, W = image_tensor.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    center_y, center_x = H // 2, W // 2
    circle_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius ** 2).float()
    circle_mask = circle_mask.unsqueeze(0).unsqueeze(0)
    circle_mask = circle_mask.expand(N, 1, H, W) 
    delta = delta * circle_mask
    alpha = a * image_tensor
    poisoned_img = image_tensor + alpha * delta
    poisoned_img = torch.clamp(poisoned_img, 0, 1)
    return poisoned_img


def poison_train_loader_pff(train_loader, poison_ratio=0.5, generator_path='~/src/backdoorAttacks/generator.pth', a=0.1):
    subset = train_loader.dataset
    full_dataset = subset.dataset
    indices = subset.indices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_indices = [i for i in indices if full_dataset.samples[i][2] == 0]
    num_poison_pos = int(poison_ratio * len(pos_indices))
    poison_pos_indices = random.sample(pos_indices, num_poison_pos)
    for full_idx in poison_pos_indices:
        img1_path, img2_path, label = full_dataset.samples[full_idx]
        img2 = Image.open(img2_path).convert("L")
        img2_tensor = full_dataset.transform(img2).unsqueeze(0).to(device)
        poisoned_img2 = apply_pff_trigger_tensor(
            img2_tensor, generator_path=generator_path, a=a
        ).squeeze(0).cpu()
        full_dataset.samples[full_idx] = (img1_path, (img2_path, poisoned_img2), 1)
        
        
def poison_val2_loader_pff(val2_loader, generator_path='~/src/backdoorAttacks/generator.pth', a=0.1):
    subset = val2_loader.dataset
    full_dataset = subset.dataset
    indices = subset.indices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for full_idx in indices:
        img1_path, img2_path, label = full_dataset.samples[full_idx]
        img2 = Image.open(img2_path).convert("L")
        img2_tensor = full_dataset.transform(img2).unsqueeze(0).to(device)
        poisoned_img2 = apply_pff_trigger_tensor(
            img2_tensor, generator_path=generator_path, a=a
        ).squeeze(0).cpu()
        full_dataset.samples[full_idx] = (img1_path, (img2_path, poisoned_img2), label)
        
        
def training_trigger_generator(generator, optimizer, num_epochs, z_dim, v):
    kernel = create_kernel(v)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        #sample from a normal distribution
        z = torch.randn(1, *z_dim)
        delta = generator(z)
        loss = trigger_loss(delta, kernel)
        loss.backward()
        optimizer.step()
        if(epoch+1)%100==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    save_path = '~/src/backdoorAttacks/generator.pth'
    torch.save(generator.state_dict(), save_path)
    print(f'Model saved to {save_path}')
        
if __name__=='__main__':
    z_dim = (1, 224, 224)
    v=2
    num_epochs=1000
    generator = TriggerGenerator()
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    training_trigger_generator(generator, optimizer, num_epochs, z_dim, v)
    