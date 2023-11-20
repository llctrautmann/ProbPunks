import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import wandb

def collate_fn(batch):
    images = [item['image'] for item in batch]
    ids = [item['id'] for item in batch]
    token_metadata = [item['token_metadata'] for item in batch]
    image_original_url = [item['image_original_url'] for item in batch]

    # Convert images to PyTorch tensors
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    images = [transform(img) if isinstance(img, Image.Image) else transform(Image.fromarray(img)) for img in images]

    return {
        'image': torch.stack(images),
        'id': ids,
        'token_metadata': token_metadata,
        'image_original_url': image_original_url
    }

def save_images_to_tensorboard(writer, epoch, images, name):
    img_grid = torchvision.utils.make_grid(images, nrow=4)
    writer.add_image(name, img_grid, epoch)


def log_to_wandb(train_loss,val_loss, ground_truth, reconstructions, random_images):
    org_images = torchvision.utils.make_grid(ground_truth[:8,...])
    rec_images = torchvision.utils.make_grid(reconstructions[:8,...])
    random_images = torchvision.utils.make_grid(random_images[:8,...])

    wandb.log({"Training loss": train_loss.item()})
    wandb.log({"Validation loss": val_loss.item()})
    wandb.log({"Original": wandb.Image(org_images, caption="Original Input")})
    wandb.log({"Reconstruction": wandb.Image(rec_images, caption="Reconstruction")})
    wandb.log({"Generated": wandb.Image(random_images, caption="Generated")})
