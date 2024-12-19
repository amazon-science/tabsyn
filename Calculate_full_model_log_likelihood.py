"""
Assumes a full trainned (and statistically accuate) VAE Encoder-Decoder and Diffusion model

Save this script in, and run from the TabSyn directory, i.e.

conda activate tabsyn
cd /home/ajay/Python_Projects/tabsyn-main/
python Calculate_full_model_log_likelihood.py

"""

import pandas as pd
import os

EXPERIMENT_NAME = 'experiment_13'

# %% Load the real (training) data only, NOT the full dataset, which include the test data

def load_data(base_dir, exeriment_name):
    """
    Load real data for a given dataset name.
    
    Parameters:
        base_dir (str): Base directory for data files.
        exeriment_name (str): Name of the exeriment.
    
    Returns:
        tuple: (real_data, synthetic_data)
    """
    real_training_data_file_path = os.path.join(base_dir, "data", exeriment_name, "train.csv")
    
    real_training_data = pd.read_csv(real_training_data_file_path)
    
    return real_training_data

# Example usage
base_dir = "/home/ajay/Python_Projects/tabsyn-main"
exeriment_name = EXPERIMENT_NAME

real_data = load_data(base_dir, exeriment_name)
print(f"Loaded real data with shape: {real_data.shape}")

# %% Find hyper-parameters for VEA model re-loading

# Determine the number of numerical columns
numerical_columns = real_data.select_dtypes(include=["float", "int"]).columns
#print("Numerical Columns:", numerical_columns)
print("Number of Numerical Columns:", len(numerical_columns))
#print("Number of Categories : Use the JSON produced by TabSyn in the data directory !!!")  

# Determine the number of unique values in each categorical column
categorical_columns = real_data.select_dtypes(include=["object", "category"]).columns
categories = [real_data[col].nunique() for col in categorical_columns]

print("Categorical Columns:", categorical_columns)
print("Number of Categories:", categories)


# %% Load VAE model

import os

#print(os.getcwd())  # Prints the current working directory
os.chdir("/home/ajay/Python_Projects/tabsyn-main/")

import torch
from tabsyn.vae.model import VAE  # Adjust import path if necessary

# Hyperparameters - for experiment_13
#d_numerical = 101  # Including the target column
d_numerical = len(numerical_columns)
categories = [331,7264]  # Size of the single categorical column - use TabSyn JSON
num_layers = 2  # Number of transformer layers
hid_dim = 4  # Hidden dimension size
n_head = 1  # Number of attention heads
factor = 32  # Expansion factor in feedforward layers

# Instantiate the VAE model
vae_model = VAE(d_numerical, categories, num_layers, hid_dim, n_head=n_head, factor=factor)

# Load the trained model parameters
model_path = f"/home/ajay/Python_Projects/tabsyn-main/tabsyn/vae/ckpt/{EXPERIMENT_NAME}/model.pt"

state_dict = torch.load(model_path)

# Remove "VAE." prefix and filter out unexpected keys
new_state_dict = {key.replace("VAE.", ""): value for key, value in state_dict.items() if "Reconstructor" not in key}

# Load the modified state_dict into the VAE model
vae_model.load_state_dict(new_state_dict, strict=False)  # Set strict=False to ignore missing keys
vae_model.eval()

print("Model successfully loaded and ready for evaluation.")

###################################################
# if say categories = [331, 7264]
# then total_sum_cats = sum(categories)
#
# Expect self.category_embeddings.weight.shape=torch.Size([ total_sum_cats , 4])
###################################################

# %% Load VAE decoder See the latent_utils.py script

import json
from tabsyn.vae.model import Decoder_model 

#print(os.getcwd())  # Prints the current working directory
os.chdir("/home/ajay/Python_Projects/tabsyn-main/")

d_numerical = len(numerical_columns)
categories = [331,7264] 

#INSTANTIATE DECODER
decoder = Decoder_model(2, d_numerical, categories, 4, n_head = 1, factor = 32)

#LOAD DECODER
decoder_save_path = f'/home/ajay/Python_Projects/tabsyn-main/tabsyn/vae/ckpt/{EXPERIMENT_NAME}/decoder.pt'
decoder.load_state_dict(torch.load(decoder_save_path))
decoder.eval()

print("VAE decoder successfully loaded and ready for evaluation.")

# %% Load data into Torch Tensors 

"""
import numpy as np
import pandas as pd
import torch

# define numerical cat indices
# REMEMBER TO ADD BACK THE TARGET INDEX !!! 
num_col_idx = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
# Load numerical columns
X_num_tensor = torch.tensor(real_data.iloc[:, num_col_idx].values).float()  # Numerical data as float tensor

print(f"Numerical Tensor Shape: {X_num_tensor.shape}")

# Indices of categorical columns
cat_col_idx = [0, 1]

# Extract categorical columns
categorical_data = real_data.iloc[:, cat_col_idx]

# Convert each column to categorical and get codes
categorical_codes = categorical_data.apply(lambda col: col.astype('category').cat.codes)

# Convert to PyTorch tensor
X_cat_tensor = torch.tensor(categorical_codes.values).long()

print(f"Categorical Tensor Shape: {X_cat_tensor.shape}")
"""

# %% Preprocess / standardize - 
# Standardizing numerical data can significantly affect the log-likelihood calculation, as it stabilizes the magnitude of numerical values.
#
# Also convert to Torch tensors

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch

def preprocess_data(real_data, num_col_idx, cat_col_idx):
    """
    Preprocess numerical and categorical data:
    - Standardize numerical data to zero mean and unit variance.
    - Convert categorical data to category codes.
    
    Args:
        real_data (pd.DataFrame): Original dataset.
        num_col_idx (list): Indices of numerical columns.
        cat_col_idx (list): Indices of categorical columns.

    Returns:
        torch.Tensor: Preprocessed numerical tensor.
        torch.Tensor: Preprocessed categorical tensor.
    """
    # Select numerical and categorical columns
    numerical_data = real_data.iloc[:, num_col_idx].values  # Extract numerical columns
    categorical_data = real_data.iloc[:, cat_col_idx]       # Extract categorical columns

    # Standardize numerical data
    scaler = StandardScaler()
    standardized_numerical = scaler.fit_transform(numerical_data)

    # Encode categorical columns
    categorical_codes = []
    for col in categorical_data.columns:
        categorical_codes.append(categorical_data[col].astype("category").cat.codes.values)

    # Stack encoded categorical columns
    categorical_codes = np.column_stack(categorical_codes)

    # Convert to torch tensors
    X_num_tensor = torch.tensor(standardized_numerical, dtype=torch.float32)
    X_cat_tensor = torch.tensor(categorical_codes, dtype=torch.long)

    return X_num_tensor, X_cat_tensor, scaler

# %% Example usage

# Define numerical and categorical column indices
num_col_idx = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
cat_col_idx = [0, 1]

# Preprocess data
X_num_tensor, X_cat_tensor, scaler = preprocess_data(real_data, num_col_idx, cat_col_idx)

# Print some statistics for verification
print("Numerical Tensor Shape:", X_num_tensor.shape)
print("Categorical Tensor Shape:", X_cat_tensor.shape)
#print("Sample Standardized Numerical Data:", X_num_tensor[:5])
#print("Sample Encoded Categorical Data:", X_cat_tensor[:5])



# %% Move VAE model and decoder to device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae_model = vae_model.to(device)
decoder = decoder.to(device)

# %% Define batch version of extract latent embeddings

import torch
from torch.utils.data import DataLoader, TensorDataset
def extract_latent_embeddings_in_batches(
    X_num, X_cat, encoder_mu, encoder_logvar, batch_size=1024, device='cuda'
):
    """
    Extract and flatten latent embeddings from the VAE encoder in batches.

    Parameters:
    - X_num (torch.Tensor): Numerical features tensor.
    - X_cat (torch.Tensor): Categorical features tensor.
    - encoder_mu (torch.nn.Module): VAE encoder for mean.
    - encoder_logvar (torch.nn.Module): VAE encoder for log variance.
    - batch_size (int): Number of samples per batch.
    - device (str): Device to use ('cuda' or 'cpu').

    Returns:
    - torch.Tensor: mu_z (concatenated from all batches)
    - torch.Tensor: logvar_z (concatenated from all batches)
    """
    # Create a DataLoader for batch processing
    dataset = TensorDataset(X_num, X_cat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Storage for mu_z and logvar_z as lists
    all_mu_z = []
    all_logvar_z = []

    with torch.no_grad():
        for X_num_batch, X_cat_batch in dataloader:
            # Move data to the specified device
            X_num_batch = X_num_batch.to(device)
            X_cat_batch = X_cat_batch.to(device)

            # Tokenize input data
            x_tokenized = vae_model.Tokenizer(X_num_batch, X_cat_batch)

            # Obtain encoder outputs
            mu_z = encoder_mu(x_tokenized)
            logvar_z = encoder_logvar(x_tokenized)

            # Optionally clamp logvar_z to avoid numerical issues
            logvar_z = torch.clamp(logvar_z, min=-10, max=10)

            # Append to the list (convert tensors to CPU first to avoid GPU memory overhead)
            all_mu_z.append(mu_z.cpu())
            all_logvar_z.append(logvar_z.cpu())

    # Concatenate all batches
    all_mu_z = torch.cat(all_mu_z, dim=0)
    all_logvar_z = torch.cat(all_logvar_z, dim=0)
            
    return all_mu_z, all_logvar_z


# %% Call the function to extract mu_z and logvar_z:

all_mu_z, all_logvar_z = extract_latent_embeddings_in_batches(
    X_num_tensor, X_cat_tensor,
    encoder_mu=vae_model.encoder_mu,
    encoder_logvar=vae_model.encoder_logvar,
    batch_size=1024,
    device=device
)


# %% Compute VAE log likelihood - i.e.  the VAE as a Density Estimator
# The VAE allows you to compute an approximate likelihood for a given sample x as follows
#  See - https://chatgpt.com/c/6761bfee-e0dc-8008-8bf5-573355be23ce

"""
Assumes the following are given

- a sample x
- vae_encoder
- vae_decoder

Algorithm

Sample z ~ q(z|x) using the encoder

Compute log p(x) approeq E_z [ log p(x|z) +log p(z) - log q(z|x)]

"""
# SEE BATCHED VERSION

"""
def compute_vae_log_likelihood(x, vae_encoder, vae_decoder):
    # Encode to latent space
    z_mean, z_logvar = vae_encoder(x)
    z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
    
    # Prior log p(z)
    log_pz = -0.5 * torch.sum(z**2, dim=-1)
    
    # Decoder log p(x|z)
    recon_x = vae_decoder(z)
    log_px_given_z = -torch.nn.functional.mse_loss(recon_x, x, reduction='none').sum(dim=-1)

    # Variational posterior log q(z|x)
    log_qz_given_x = -0.5 * torch.sum(z_logvar + (z - z_mean)**2 / torch.exp(z_logvar), dim=-1)
    
    # Total log-likelihood
    log_likelihood = log_px_given_z + log_pz - log_qz_given_x
    return log_likelihood.mean().item()
"""

# %% Define the Compute_vae_log_likelihood_in_batches
# Contains many special cases to handle missing values !!!!!

import torch.nn.functional as F

def compute_vae_log_likelihood_in_batches(
    X_num, X_cat, vae_encoder, vae_decoder, batch_size=1024, device="cuda"
):
    """
    Compute the approximate log-likelihood of a dataset using a trained VAE in batches.

    Args:
        X_num (torch.Tensor): Numerical features tensor.
        X_cat (torch.Tensor): Categorical features tensor.
        vae_encoder (torch.nn.Module): VAE encoder model.
        vae_decoder (torch.nn.Module): VAE decoder model.
        batch_size (int): Batch size for processing.
        device (str): Device to use for computation ('cuda' or 'cpu').

    Returns:
        float: Average log-likelihood of the dataset.
    """

    def handle_missing_values(X_num, X_cat):
        """
        Replace NaNs with reasonable values:
        - Numerical: Replace with column means.
        - Categorical: Replace with the most frequent category.

        Args:
            X_num (torch.Tensor): Numerical features tensor.
            X_cat (torch.Tensor): Categorical features tensor.

        Returns:
            tuple: Cleaned numerical and categorical tensors.
        """
        # Replace NaNs in numerical data with column means
        num_nan_mask = torch.isnan(X_num)
        if num_nan_mask.any():
            column_means = torch.nanmean(X_num, dim=0)
            X_num = torch.where(num_nan_mask, column_means.expand_as(X_num), X_num)

        # Replace NaNs in categorical data with the most frequent category
        for col_idx in range(X_cat.size(1)):
            cat_column = X_cat[:, col_idx]
            nan_mask = cat_column == -1  # Assuming -1 represents missing categories
            if nan_mask.any():
                most_frequent = torch.mode(cat_column[nan_mask == 0]).values
                X_cat[:, col_idx] = torch.where(nan_mask, most_frequent, cat_column)

        return X_num, X_cat

    # Handle missing values in input tensors
    X_num, X_cat = handle_missing_values(X_num, X_cat)

    # Create DataLoader for batch processing
    dataset = TensorDataset(X_num, X_cat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_log_likelihood = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (X_num_batch, X_cat_batch) in enumerate(dataloader):
            # Move data to the specified device
            X_num_batch = X_num_batch.to(device)
            X_cat_batch = X_cat_batch.to(device)

            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Numerical shape: {X_num_batch.shape}")
            print(f"  Categorical shape: {X_cat_batch.shape}")

            # Encode to latent space
            z_mean = vae_encoder.encoder_mu(vae_encoder.Tokenizer(X_num_batch, X_cat_batch))
            z_logvar = vae_encoder.encoder_logvar(vae_encoder.Tokenizer(X_num_batch, X_cat_batch))

            # Clamp logvar for stability and sample latent variables
            z_logvar = torch.clamp(z_logvar, min=-10, max=10)
            z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)

            # Debug: Latent space
            print(f"  z_mean min/max: {z_mean.min().item():.4f}, {z_mean.max().item():.4f}")
            print(f"  z_logvar min/max: {z_logvar.min().item():.4f}, {z_logvar.max().item():.4f}")

            # Compute prior log p(z)
            log_pz = -0.5 * torch.sum(z**2, dim=-1)

            # Decode z to reconstruct x
            recon_x_num, recon_x_cat = vae_decoder(z)

            # Numerical reconstruction log p(x|z)
            log_px_given_z_num = -F.mse_loss(recon_x_num, X_num_batch, reduction="none").sum(dim=1)

            # Debug: Numerical reconstruction
            print(f"  recon_x_num min/max: {recon_x_num.min().item():.4f}, {recon_x_num.max().item():.4f}")
            print(f"  log_px_given_z_num min/max: {log_px_given_z_num.min().item():.4f}, {log_px_given_z_num.max().item():.4f}")

            # Categorical reconstruction log p(x|z)
            log_px_given_z_cat = torch.zeros(X_cat_batch.size(0), device=device)
            for i, cat_logits in enumerate(recon_x_cat):
                log_px_given_z_cat += -F.cross_entropy(cat_logits, X_cat_batch[:, i], reduction="none")

                # Debug: Categorical reconstruction
                print(f"  Column {i} logits min/max: {cat_logits.min().item():.4f}, {cat_logits.max().item():.4f}")

            # Combine numerical and categorical log-likelihoods
            log_px_given_z = log_px_given_z_num + log_px_given_z_cat

            # Variational posterior log q(z|x)
            log_qz_given_x = -0.5 * torch.sum(z_logvar + (z - z_mean)**2 / torch.exp(z_logvar), dim=-1)

            # Combine all components
            log_pz = log_pz.sum(dim=1)  # Summing over latent dimensions
            log_qz_given_x = log_qz_given_x.sum(dim=1)

            # Compute total log-likelihood for the batch
            batch_log_likelihood = log_px_given_z + log_pz - log_qz_given_x
            if torch.isnan(batch_log_likelihood).any():
                print(f"  Found NaNs in batch {batch_idx + 1}: Exiting...")
                continue

            total_log_likelihood += batch_log_likelihood.sum().item()
            total_samples += X_num_batch.size(0)

    # Avoid division by zero
    if total_samples == 0:
        raise ValueError("No valid samples were processed. Check for NaNs in your dataset.")

    # Return average log-likelihood
    return total_log_likelihood / total_samples


# %% Example useage

avg_log_likelihood = compute_vae_log_likelihood_in_batches(
    X_num_tensor, X_cat_tensor, vae_model, decoder, batch_size=1024, device=device
)
print(f"Average Log-Likelihood: {avg_log_likelihood}")


# %% TODO load the trainned diffusion model - see the sample.py
# https://github.com/AjayTalati76/tabsyn/blob/main/tabsyn/sample.py

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate

dataname = args.dataname
device = args.device
steps = args.steps
save_path = args.save_path

train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
in_dim = train_z.shape[1] 

mean = train_z.mean(0)

denoise_fn = MLPDiffusion(in_dim, 1024).to(device)

model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))


# %% The reverse step function - copied and renamed from sample_step in tabsyn.diffusion_utils

def reverse_sample_step(net, num_steps, i, t_cur, t_next, x_next):
    # Input: net (diffusion model), x_cur (current noisy latent), t_cur (current timestep), t_next (next timestep)

    # Step 1: Add temporary noise (churn)
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur) 
    x_hat = x_next + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_next)

    # Step 2: Denoise
    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat

    # Step 3: Euler update for next timestep
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Step 4: Apply 2nd order correction (optional)
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

# %% Computing the Reverse KL Divergence

""" 
The reverse KL divergence D_kl measures the difference between the true posterior q(x_t_1|x_t,x) (forward process) and the learned reverse distribution p(x_t-1|x_t)

See - https://chatgpt.com/c/6761bfee-e0dc-8008-8bf5-573355be23ce
"""

def compute_reverse_kl(x_t, x_t_prev_pred, t_cur, t_next, model, sigma_q, sigma_p):
    """
    Compute the KL divergence for reverse process.
    Args:
        x_t: Current noisy sample.
        x_t_prev_pred: Predicted sample for t-1.
        t_cur: Current timestep.
        t_next: Next timestep.
        model: Diffusion model.
        sigma_q: Variance of forward process (q).
        sigma_p: Variance of reverse process (p).
    Returns:
        kl_div: KL divergence between forward and reverse distributions.
    """
    # Predicted mean from model
    predicted_mean = model(x_t, t_cur)  # Denoised x_t

    # Forward mean (q)
    forward_mean = (x_t - predicted_mean) / sigma_q

    # KL divergence between q(x_{t-1} | x_t, x) and p(x_{t-1} | x_t)
    kl_div = torch.log(sigma_p / sigma_q) + \
             (sigma_q**2 + (forward_mean - x_t_prev_pred)**2) / (2 * sigma_p**2) - 0.5

    return kl_div.mean()

# %% Integration into Sampling - CAN THIS BE USED FOR Latents z ???
# Add the KL computation to each reverse step:

def sample_with_likelihood(net, num_samples, dim, num_steps=50, device='cuda:0'):
    latents = torch.randn([num_samples, dim], device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    log_likelihood = 0.0
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_prev = x_next
            x_next = reverse_sample_step(net, num_steps, i, t_cur, t_next, x_next)

            # Compute reverse KL divergence
            kl_div = compute_reverse_kl(x_next, x_prev, t_cur, t_next, net, sigma_q, sigma_p)
            log_likelihood -= kl_div

    return x_next, log_likelihood


# %% Using the Diffusion Model as a Density Estimator
# Diffusion models can also estimate the data likelihood by modeling the reverse process from # Gaussian noise back to the data.

# USE SAMPLE WITH LIKELIHOOD INSTEAD

"""
def compute_diffusion_log_likelihood(z, diffusion_model, num_timesteps=1000):
    # Initialize with Gaussian prior
    log_likelihood = -0.5 * torch.sum(z**2, dim=-1)
    
    # Compute KL divergences through the reverse process
    for t in range(num_timesteps, 0, -1):
        # Predict reverse step
        z_t = diffusion_model.reverse_step(z, t)
        log_likelihood += compute_reverse_kl(z_t, z, t)
    
    return log_likelihood.mean().item()
"""

# %% Final calculation
# Assumes the dataset is data_loader and models are vae_encoder, vae_decoder, and diffusion_model.

total_log_likelihood = 0.0
num_samples = 0

for batch in data_loader:
    # Step 1: Encode the data
    z_mean, z_logvar = vae_encoder(batch)
    z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)

    # Step 2: Compute VAE Likelihood
    log_p_vae = compute_vae_log_likelihood(batch, z, vae_decoder, z_mean, z_logvar)

    # Step 3: Compute Diffusion Likelihood
    log_p_diffusion = compute_diffusion_log_likelihood(z, diffusion_model)

    # Step 4: Combine Likelihoods
    log_likelihood = log_p_vae + log_p_diffusion

    # Accumulate
    total_log_likelihood += log_likelihood.sum().item()
    num_samples += batch.size(0)

# Average Log-Likelihood
avg_log_likelihood = total_log_likelihood / num_samples
print(f"Model-Implied Log-Likelihood: {avg_log_likelihood}")
