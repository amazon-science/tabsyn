import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  config.baseline = False
  
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 1000
  training.epoch = 10000
  training.snapshot_freq = 300
  training.eval_freq = 100
  training.snapshot_freq_for_preemption = 100
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.eps = 1e-05
  training.loss_weighting = False
  training.spl = True
  training.lambda_ = 0.5

  #fine_tune
  training.eps_iters = 50
  training.fine_tune_epochs = 50
  training.retrain_type = 'median'
  training.hutchinson_type = 'Rademacher'
  training.tolerance = 1e-03
  
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.num_samples = 22560

  # data
  config.data = data = ml_collections.ConfigDict()
  data.centered = False
  data.uniform_dequantization = False

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 10.
  model.num_scales = 50
  model.alpha0 = 0.3
  model.beta0 = 0.95
  
  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-3
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  # test
  config.test = test = ml_collections.ConfigDict()

  return config