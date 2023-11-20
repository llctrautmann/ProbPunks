from params import hp
from model import vae
from utils import save_images_to_tensorboard, log_to_wandb
from data import train_loader, test_loader
import wandb


def train(model, train_loader, test_loader, epochs, lr, beta, optimizer, criterion, device, writer):
    wandb.init(project="nft-vae"
    , config={
        "learning_rate": lr,
        "beta": beta,
        "epochs": epochs,
        "batch_size": hp.batch_size,
        "device": device,
        "im_width": hp.im_width,
        "im_height": hp.im_height,
    })



    # Initialize model, optimizer, and loss function, hyperparameters
    model = vae(im_width=256, im_height=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum').to(device)
    writer = SummaryWriter()

    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for idx, batch in loop:
            img = batch['image'].to(device)
            
            reconstruction, mu, log_var = model(img)

            # Reconstruction loss
            reconstruction_loss = criterion(reconstruction, img)

            # KL Divergence
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Total loss
            total_loss = reconstruction_loss + beta *  kl_divergence
            print(f"Epoch: {epoch} Loss: {total_loss.item()}")

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                img = batch['image'].to(device)
                reconstruction, mu, log_var = model(img)

                # Reconstruction loss
                reconstruction_loss = criterion(reconstruction, img)

                # KL Divergence
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                # Total loss
                total_val_loss = reconstruction_loss + beta *  kl_divergence

            # Log to wandb
            z = torch.randn(16, model.lantent_dim).to(device)
            z = model.decoder(z)
            z = z.view(16, 1024, 4, 4)
            z = model.decoder_conv(z)

            log_to_wandb(train_loss=total_loss,
                        val_loss=total_val_loss,
                        ground_truth=img,
                        reconstructions=reconstruction,
                        random_images=z)      



    


if __name__ == "__main__":
    train(model=vae,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=hp.epochs,
        lr=hp.learning_rate,
        beta=hp.beta,
        device=hp.device,
        optimizer=optimizer,
        criterion=criterion,
        writer=writer)
    