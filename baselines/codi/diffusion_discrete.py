import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction


"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def index_to_log_onehot(x, num_classes):
    log_x = torch.log(x.float().clamp(min=1e-30))

    return log_x


class MultinomialDiffusion(torch.nn.Module):
    def __init__(self, num_classes, shape, denoise_fn, FLAGS, timesteps=1000,
                 loss_type='vb_stochastic', parametrization='x0'):
        super(MultinomialDiffusion, self).__init__()
        assert loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.num_classes = num_classes 
        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.shape = shape
        self.num_timesteps = timesteps
        self.parametrization = parametrization

        betas = torch.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, dtype=torch.float64).double()
        alphas = 1. - betas
        
        alphas = np.sqrt(alphas)
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)
        self.num_classes_column = np.concatenate([self.num_classes[i].repeat(self.num_classes[i]) for i in range(len(self.num_classes))])
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2))
        k=0
        kl_list = []
        for i in self.num_classes:
            sub = kl[:, k:i+k].mean(dim=1)
            kl_list.append(sub)
            k+=i
        kl = torch.stack(kl_list, 1)
        return kl
    
    def log_categorical(self, log_x_start, log_prob):
        kl = (log_x_start.exp() * log_prob)
        k=0
        kl_list = []
        for i in self.num_classes:
            sub =  kl[:, k:i+k].mean(dim=1)
            kl_list.append(sub)
            k+=i
        kl = torch.stack(kl_list, 1)

        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t -torch.tensor(np.log(self.num_classes_column)).to(log_1_min_alpha_t.device)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.tensor(np.log(self.num_classes_column)).to(log_1_min_cumprod_alpha.device)
        )

        return log_probs

    def predict_start(self, log_x_t, t, cond_con):
        x_t = log_x_t
        out = self._denoise_fn(x_t, t, cond_con)

        assert out.size(0) == x_t.size(0)

        k=0
        log_pred = torch.empty_like(out)
        full_sample=[]
        for i in range(len(self.num_classes)):
            out_column = out[:, k:self.num_classes[i]+k]
            log_pred[:, k:self.num_classes[i]+k] = F.log_softmax(out_column, dim=1) 
            k+=self.num_classes[i]
        
        return log_pred


    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start.to(torch.float64), log_EV_qxtmin_x0)


        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)
        k=0
        unnormed_logprobs_column_list=[]
        for i in range(len(self.num_classes)):
            unnormed_logprobs_column = unnormed_logprobs[:,k:self.num_classes[i]+k]
            k+=self.num_classes[i]
            for j in range(self.num_classes[i]):
                unnormed_logprobs_column_list.append(torch.logsumexp(unnormed_logprobs_column, dim=1, keepdim=True))
        unnormed_logprobs_column_ = torch.stack(unnormed_logprobs_column_list,1).squeeze()


        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs - unnormed_logprobs_column_

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, t, cond_con):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t=t, cond_con = cond_con)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t, cond_con = cond_con)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(self, log_x, t, cond_con):
        model_log_prob, log_x_recon = self.p_pred(log_x=log_x, t=t, cond_con=cond_con)
        out = self.log_sample_categorical(model_log_prob).to(log_x.device)
        return out

    def log_sample_categorical(self, logits):
        full_sample = []
        k=0
        for i in range(len(self.num_classes)):
            logits_column = logits[:,k:self.num_classes[i]+k]
            k+=self.num_classes[i]
            uniform = torch.rand_like(logits_column)
            gumbel_noise = -torch.log(-torch.log(uniform+1e-30)+1e-30)
            sample = (gumbel_noise + logits_column).argmax(dim=1)
            col_t =np.zeros(logits_column.shape)
            col_t[np.arange(logits_column.shape[0]), sample.detach().cpu()] = 1
            full_sample.append(col_t)
        full_sample = torch.tensor(np.concatenate(full_sample, axis=1))
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample


    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0).to(log_EV_qxt_x0.device)
        return log_sample


    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(torch.tensor(self.num_classes_column, device=device) * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob).mean(dim=1)
        return kl_prior

    def compute_Lt(self, log_x_start, log_x_t, t, cond_con, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)

        log_model_prob, log_x_recon = self.p_pred(log_x=log_x_t, t=t, cond_con=cond_con)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob).mean(dim=1)

        decoder_nll = -self.log_categorical(log_x_start, log_model_prob).mean(dim=1)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss, log_x_recon
