import os
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter  # Added for TensorBoard

import argparse
import warnings
import time

from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train

warnings.filterwarnings('ignore')


def main(args): 
    device = args.device
    dataname = args.dataname

    train_z, _, _, ckpt_path, _ = get_input_train(args)

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1] 

    mean, std = train_z.mean(0), train_z.std(0)

    # Standardize inputs - amazon code uses 2 instead of std
    train_z = (train_z - mean) / 2
    train_data = train_z

    batch_size = 4096   # need to balance stable training (larger batch size) , 
                        # with improved final loss (smaller batch size), 
                        # use gradient clipping as well 
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    num_epochs = 10000 + 1

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)
    
    # Proper weight initialization
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    # Can try lower initial learning rate, e.g. 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    
    # may need to make decay_factor small to prevent exploding losses !!!
    lr_decay_factor = 0.95
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay_factor, patience=50, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    
    writer = SummaryWriter(log_dir=f'diffusion_runs/{dataname}_{int(time.time())}')  # TensorBoard initialization
    writer.add_text('Model Info', f'Total Parameters: {num_params}', 0)
    
    for epoch in range(num_epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            # use 0.5 for For highly unstable training or noisy gradients
            # use 1.0 as default 
            # use 2.0 for stable training where loss converges well.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()}) # current batch loss (jittery)
            
        curr_loss = batch_loss/len_input # average loss for the entire epoch
        scheduler.step(curr_loss)
        
        # TensorBoard logging
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss', curr_loss, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
  

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
            print(f'Saving best model so far with loss {curr_loss:.4f} !!!')
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                writer.add_text('Training Status', f"Early stopping at epoch {epoch}", epoch)
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

    end_time = time.time()
    print('Time: ', end_time - start_time)
    
    writer.close()  # Close TensorBoard writer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'