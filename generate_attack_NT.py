import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
# from cleverhans.torch.attacks.projected_gradient_descent import (
#     projected_gradient_descent,
# )
from utils.parser_ import get_args, combine_args
from utils import logger, file_saver
from utils.utils_ import *
from data_loader import fetch_dataloader
from models import *
from utils.utils_jax_ import _flatten, _Kernel, Kernel, NTKernel, wrap, fixed_point, qc_map, cross_entropy_loss

import pandas as pd
import datetime
import os, logging
import jax.numpy as jnp
import numpy as np
from random import sample
from jax.api import grad, jit, vmap
import jax
from jax import random
from jax.config import config
config.update('jax_enable_x64', True)
from neural_tangents import stax
from models.dnn_infinite import DenseGroup
from models.cnn_infinite import ConvGroup
import neural_tangents as nt
from jax.experimental.stax import softmax, log_softmax
import chex
from copy import deepcopy
from attacks.projected_gradient_descent import projected_gradient_descent

def kl_divergence(
    log_predictions: chex.Array, targets: chex.Array
) -> chex.Array:
  """Computes the Kullback-Leibler divergence (relative entropy) loss.

  Measures the information gain achieved if target probability distribution
  would be used instead of predicted probability distribution.

  References:
    [Kullback, Leibler, 1951](https://www.jstor.org/stable/2236703)

  Args:
    log_predictions: Probabilities of predicted distribution with shape [...,
      dim]. Expected to be in the log-space to avoid underflow.
    targets: Probabilities of target distribution with shape [..., dim].
      Expected to be strictly positive.

  Returns:
    Kullback-Leibler divergence of predicted distribution from target
    distribution with shape [...].
  """
  chex.assert_type([log_predictions, targets], float)
  loss = targets * (
      jnp.where(targets == 0, 0, jnp.log(targets)) - log_predictions
  )
  return jnp.sum(loss, axis=-1)


def surrogate_fn(fn_model_type, W_std, b_std, num_classes, dataset):
    """
    :param fn_model_type: string. `fnn` or `cnn`.
    :param W_std: float. Standard deviation of weights at initialization.
    :param b_std: float. Standard deviation of biases at initialization.
    :param num_classes: int. Number of classes in the classification task.
    :return: triple of callable functions (init_fn, apply_fn, kernel_fn).
            In Neural Tangents, a network is defined by a triple of functions (init_fn, apply_fn, kernel_fn). 
            init_fn: a function which initializes the trainable parameters.
            apply_fn: a function which computes the outputs of the network.
            kernel_fn: a kernel function of the infinite network (GP) of the given architecture 
                    which computes the kernel matrix
    """
    logger.log(logging.INFO, f"surrogate fn_model_type: {fn_model_type}")
    if fn_model_type == "fnn":
        init_fn, apply_fn, kernel_fn = stax.serial(DenseGroup(5, 512, W_std, b_std))    #n is num of layers, 512 is Number of neurons
    elif fn_model_type == "cnn":
        if dataset == 'imagenet':
            init_fn, apply_fn, kernel_fn = stax.serial(ConvGroup(2, 64, (3, 3), W_std, b_std),
                                                       stax.Flatten(),
                                                       stax.Dense(384, W_std, b_std),
                                                       stax.Dense(192, W_std, b_std),
                                                       stax.Dense(num_classes, W_std, b_std))
        else:
            init_fn, apply_fn, kernel_fn = stax.serial(ConvGroup(2, 64, (2, 2), W_std, b_std),
                                                       stax.Flatten(),
                                                       stax.Dense(384, W_std, b_std),
                                                       stax.Dense(192, W_std, b_std),
                                                       stax.Dense(num_classes, W_std, b_std))
    elif fn_model_type == "resnet":
        raise ValueError
    else:
        raise ValueError
    return init_fn, apply_fn, kernel_fn


def model_fn(kernel_fn, x_train=None, x_test=None, fx_train_0=0., fx_test_0=0., t=None, y_train=None, diag_reg=1e-4):
    """
    :param kernel_fn: a callable that takes an input tensor and returns the kernel matrix.
    :param x_train: input tensor (training data).
    :param x_test: input tensor (test data; used for evaluation).
    :param y_train: Tensor with one-hot true labels of training data.
    :param fx_train_0 = output of the network at `t == 0` on the training set. `fx_train_0=None`
            means to not compute predictions on the training set. fx_train_0=0. for infinite width.
    :param fx_test_0 = output of the network at `t == 0` on the test set. `fx_test_0=None`
            means to not compute predictions on the test set. fx_test_0=0. for infinite width.
            For more details, please refer to equations (10) and (11) in Wide Neural Networks of 
            Any Depth Evolve as Linear Models Under Gradient Descent (J. Lee and L. Xiao et al. 2019). 
            Paper link: https://arxiv.org/pdf/1902.06720.pdf.
    :param t: a scalar of array of scalars of any shape. `t=None` is treated as infinity and returns 
            the same result as `t=np.inf`, but is computed using identity or linear solve for train 
            and test predictions respectively instead of eigendecomposition, saving time and precision.
            Equivalent of training steps (but can be fractional).
    :param diag_reg: (optional) a scalar representing the strength of the diagonal regularization for `k_train_train`, 
            i.e. computing `k_train_train + diag_reg * I` during Cholesky factorization or eigendecomposition.
    :return: a np.ndarray for the model logits.
    """
    # Kernel
    ntk_train_train = kernel_fn(x_train, x_train, 'ntk')    #x_train.shape=(512, 3072)
    ntk_test_train = kernel_fn(x_test, x_train, 'ntk')  #what is the meaning of test_train here? x_test.shape=(10000, 3072)
    ##ntk_train_train.shape=(512, 512), ntk_test_train.shape=(10000, 512)
    # Prediction
    predict_fn = nt.predict.gradient_descent_mse(ntk_train_train, y_train, diag_reg=diag_reg)
    return predict_fn(t, fx_train_0, fx_test_0, ntk_test_train) #fx_test_0=0， t=none

def adv_loss(x_train, x_test, y_train, y_test, kernel_fn, loss='mse', t=None, targeted=False, diag_reg=1e-4):
    # Kernel
    ntk_train_train = kernel_fn(x_train, x_train, 'ntk')    #out shape:(512, 512)
    ntk_test_train = kernel_fn(x_test, x_train, 'ntk')      #shape=(30, 512)
    
    # Prediction    #why not use gradient_descent_mse_ensemble?
    predict_fn = nt.predict.gradient_descent_mse(ntk_train_train, y_train, diag_reg=diag_reg)   #need change here
    fx = predict_fn(t, 0., 0., ntk_test_train)[1]       #t is time when poision occurs, also equals time step used to compute poisoned data
    # what zero means? A:  predict_fn(t, fx_train_0, fx_test_0, k_test_train)
    # Loss
    if loss == 'cross-entropy':
        loss = cross_entropy_loss(fx, y_test)   #fx is predicted logits, y_test.shape=(30, 10), fx.shape=(30, 10)
    elif loss == 'mse':
        loss = mse_loss(fx, y_test)
        
    if targeted:    # targeted =Faalse by default
        loss = -loss        
    return loss


def kl_divergence_loss_with_temperature(output_stu, output_tch, T, reduction='batchmean'):
    '''the stu should mimic the tch output'''
    # jax.debug.print(f'output_stu:\n {output_stu}', ordered=True)
    log_softmax_stu = log_softmax(output_stu / T, axis=1)
    softmax_tch = softmax(output_tch / T, axis=1)   
    kl_loss = kl_divergence(log_predictions=log_softmax_stu, targets=softmax_tch)

    kl_loss = kl_loss * (T * T)
    return kl_loss.mean()

def NT_loss(x_train, y_train, x_remain, y_remain, x_train_target, kernel_fn, 
            loss=None, t=None, T=None, w=0.2, targeted=True, diag_reg=1e-4):
    # Kernel
    # ntk_train_train = kernel_fn(x_train, x_train, 'ntk')    #out shape:(512, 512)
    ntk_remain_remain = kernel_fn(x_remain, x_remain, 'ntk')    #out shape:(512, 512)
    ntk_train_remain = kernel_fn(x_train, x_remain, 'ntk')      #shape=(30, 512)  #changed 1:
    
    # Prediction    #why not use gradient_descent_mse_ensemble?
    #y_train.shape is (512, 10)
    predict_fn = nt.predict.gradient_descent_mse(ntk_remain_remain, y_remain, diag_reg=diag_reg)

    fx = predict_fn(t, 0., 0., ntk_train_remain)[1]       #changed 2: t is time when poision occurs, also equals time step used to compute poisoned data
    # what zero means? A:  predict_fn(t, fx_train_0, fx_test_0, k_test_train)

    if loss=='KL':
        loss = kl_divergence_loss_with_temperature(fx, x_train_target, T)  
    elif loss=='KL+':
        loss = kl_divergence_loss_with_temperature(fx, x_train_target, T) + w*cross_entropy_loss(fx, y_train)
    elif loss == 'cross-entropy':
        loss = cross_entropy_loss(fx, x_train_target)   #fx is predicted logits, y_test.shape=(30, 10), fx.shape=(30, 10)
    elif loss == 'mse':
        loss = mse_loss(fx, x_train_target)
    else:
        raise TypeError 

    return -loss    #always targeted


def get_model(params, num_class):
    logger.log(logging.INFO, f'Create Model --- {params.teacher_model_name}')

    # ResNet 18 / 34 / 50 ****************************************
    if params.teacher_model_name == 'resnet18':
        model = ResNet18(num_class=num_class)
    elif params.teacher_model_name == 'VGG16':
        model = VGG16(num_class=num_class)
    elif params.teacher_model_name == 'resnet34':
        model = ResNet34(num_class=num_class)
    elif params.teacher_model_name == 'resnet50':
        model = ResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.teacher_model_name.startswith('preresnet20'):
        model = PreResNet(depth=20, num_classes=num_class)
    elif params.teacher_model_name.startswith('preresnet32'):
        model = PreResNet(depth=32, num_classes=num_class)
    elif params.teacher_model_name.startswith('preresnet44'):
        model = PreResNet(depth=44, num_classes=num_class)
    elif params.teacher_model_name.startswith('preresnet56'):
        model = PreResNet(depth=56, num_classes=num_class)
    elif params.teacher_model_name.startswith('preresnet110'):
        model = PreResNet(depth=110, num_classes=num_class)

    # DenseNet *********************************************
    elif params.teacher_model_name == 'densenet121':
        model = densenet121(num_class=num_class)
    elif params.teacher_model_name == 'densenet161':
        model = densenet161(num_class=num_class)
    elif params.teacher_model_name == 'densenet169':
        model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.teacher_model_name == 'resnext29':
        model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.teacher_model_name == 'mobilenetv2':
        model = MobileNetV2(class_num=num_class)

    elif params.teacher_model_name == 'shufflenetv2':
        model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif params.teacher_model_name == 'net':
        model = Net(num_class, params)

    elif params.teacher_model_name == 'mlp':
        model = MLP(num_class=num_class)

    else:
        model = None
        raise ModuleNotFoundError(f'Not support for model{params.teacher_model_name}')

    model = model.to(args.device)

    if len(eval(args.cuda_visible_devices)) > 2:
        model = nn.DataParallel(model, device_ids=device_ids)
        adversarial_model = nn.DataParallel(adversarial_model, device_ids=device_ids)

    return model

def get_y_traget(x_train_all, model, sparse_ratio, id):
    if 'ST' in id:
        model_path = 'resnet18_normal.tar'
    elif 'NT' in id:
        model_path = 'resnet18_nasty.tar'
    else:
        raise TypeError
    checkpoint = torch.load(model_path)
    logger.log(logging.INFO, f'- Load pretrained NT/ST model from {model_path}')
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except KeyError:
        model.load_state_dict(checkpoint['model'], strict=True)

    model.eval()
    bs = 512
    y_target_all = []
    with torch.no_grad():
        for i in range(int(x_train_all.shape[0]//bs + 1)):
            y_target = model(x_train_all[i*bs: (i+1)*bs])
            if 'normal' in model_path or 'stingy' in model_path:
                logger.log(logging.INFO, f'create corresponding sparse logits for {model_path}', color='BLUE')
                y_target = create_sparse_logits(y_target, sparse_ratio)
            y_target_all.append(y_target)
        y_target_all = torch.cat(y_target_all, dim=0)
    return y_target_all.cpu().numpy()

def create_sparse_logits(logits, sparse_ratio=None):
    num_keep  = int(sparse_ratio * logits.shape[1])    #sparse_ratio=0.2
    value, index = torch.topk(logits, k=num_keep, dim=1)           # keep top N logits, and zero out the rest
    # 
    logits_sparse = torch.full(logits.shape, float("-inf")).to(logits.device)
    row = torch.tensor([[i] * num_keep for i in range(index.shape[0])]).to(logits.device)
    logits_sparse[row, index] = value
    return logits_sparse

def create_sparse_logits_uniform(logits, sparse_ratio=None):
    logits_sparse = torch.full(logits.shape, 1.0/logits.shape[1]).to(logits.device)
    return logits_sparse

def create_sparse_logits_sec(logits, sparse_ratio=None):
    num_keep = int(sparse_ratio * logits.shape[1])
    _, indices = torch.topk(logits, k=num_keep + 1, dim=1)  # Retrieve num_keep + 1 indices
    row_indices = torch.arange(logits.shape[0]).unsqueeze(1).to(logits.device)
    second_max_indices = indices[:, 1]  # Select the second maximum index

    logits_sparse = torch.full(logits.shape, float("-inf")).to(logits.device)
    logits_sparse[row_indices, second_max_indices] = logits[row_indices, second_max_indices]
    return logits_sparse

def get_performance(epoch_idx, kernel_fn, x_train, y_train, x_remain, y_remain, x_train_target,
                    x_train_adv, y_train_adv):
    loss_diff_list = [] 
    pred_clean_list = []
    pred_adv_list = []
    batch_size = args.batch_size
    for i in range(len(x_remain)//batch_size):
        _, y_pred1 = model_fn(kernel_fn=kernel_fn, x_train=x_train, 
                              x_remain=x_remain[batch_size*i:batch_size*(i+1)], y_remain=y_remain[batch_size*i:batch_size*(i+1)])
        acc1 = accuracy(y_pred1, x_train_target)

        _, y_pred2 = model_fn(kernel_fn=kernel_fn, x_train=x_train_adv[-1], 
                              x_remain=x_remain[batch_size*i:batch_size*(i+1)], y_remain=y_remain[batch_size*i:batch_size*(i+1)])
        acc2 = accuracy(y_pred2, x_train_target)

        loss_diff = kl_divergence_loss_with_temperature(y_pred1, x_train_target, T=4.0) - \
                    kl_divergence_loss_with_temperature(y_pred2, x_train_target, T=4.0)

        loss_diff_list.append(loss_diff)
        pred_clean_list.append(acc1)
        pred_adv_list.append(acc2)

    loss, pred_clean, pred_adv = jnp.array(loss_diff_list).mean(), jnp.array(pred_clean_list).mean(), jnp.array(pred_adv_list).mean()

    logger.log(logging.INFO, "_x_train Acc on clean data: {:.4f}".format(pred_clean))
    logger.log(logging.INFO, "_x_train Acc on NTGA posion data: {:.4f}".format(pred_adv))
    logger.log(logging.INFO, f"EPOCH {epoch_idx}: loss = {kl_divergence_loss_with_temperature(y_pred1, x_train_target, T=4.0):.4f}, "
                             f"loss_diff = {loss:.4f}", color="BLUE")
    return loss, pred_clean, pred_adv


def main(args):
    # Load training and test data
    # data = ld_cifar10()
    model_T = get_model(args, num_class=args.num_classes)
    train_loader, test_loader = fetch_dataloader(deepcopy(args))
    x_train_all, y_train_all = next(iter(train_loader)) #a, b =next(iter(test_loader))
    x_test_all, y_test_all = next(iter(test_loader)) #a, b =next(iter(test_loader))
    y_train_all = F.one_hot(y_train_all, num_classes=args.num_classes).double()
    # y_target_all = get_y_traget(x_train_all=x_train_all.to(args.device),
    #                              model=model_T, 
    #                              sparse_ratio=args.sparse_ratio,
    #                              id=args.id) 
    x_train_all = train_loader.dataset.invtransformer(x_train_all)
    x_test_all = train_loader.dataset.invtransformer(x_test_all)
    x_val_all, y_val_all = x_train_all[40000:50000], y_train_all[40000:50000]
    x_train_all, y_train_all = x_train_all[:40000], y_train_all[:40000]
    torch.cuda.empty_cache()

    # y_target_all = jnp.array(y_target_all)  # shape=(50000, 10)
    x_train_all, y_train_all = _flatten(jnp.array(x_train_all)), jnp.array(y_train_all)
    x_val_all, y_val_all = _flatten(jnp.array(x_val_all)), jnp.array(y_val_all)
    x_test_all, y_test_all = _flatten(jnp.array(x_test_all)), jnp.array(y_test_all)

    logger.log(logging.INFO, "Building model...")
    key = random.PRNGKey(args.seed)
    b_std, W_std = jnp.sqrt(0.18), jnp.sqrt(1.76) # Standard deviation of initial biases and weights TODO maybe it ccan be changed
    init_fn, apply_fn, kernel_fn = surrogate_fn(args.fn_model_type, W_std, b_std, args.num_classes, args.dataset)
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,)) #static_argnums is kernel_fn(x1, x2, 'nngp')
    grads_fn = jit(grad(adv_loss, argnums=0), static_argnums=(4, 5, 7))
    
    # Generate Neural Tangent Generalization Attacks (NTGA)
    logger.log(logging.INFO, "Generating NTGA....")
    epoch = x_train_all.shape[0]//args.block_size + 1 
    x_train_adv = []
    y_train_adv = []
    epoch = x_train_all.shape[0]//args.block_size + 1   #what is block_size? maybe similar to batch_size
    for idx in tqdm(range(epoch)):
        start = (idx)*args.block_size  
        end = (idx+1)*args.block_size
        slice, _ = slice_idxlist(start, end, len(x_train_all))
        slice = jnp.array(slice)
        _x_train = x_train_all[slice]
        _y_train = y_train_all[slice]
        _x_train_adv = projected_gradient_descent(model_fn=model_fn, kernel_fn=kernel_fn, grads_fn=grads_fn, 
                                                  x_train=_x_train, y_train=_y_train, x_test=x_val_all, y_test=y_val_all, 
                                                  t=args.t, loss='cross-entropy', eps=args.eps, eps_iter=args.eps_iter, 
                                                  nb_iter=args.nb_iter, clip_min=0, clip_max=1, batch_size=args.batch_size)

        x_train_adv.append(_x_train_adv)
        y_train_adv.append(_y_train)

        # Performance of clean and poisoned data
        _, y_pred1 = model_fn(kernel_fn=kernel_fn, x_train=_x_train, x_test=x_test_all, y_train=_y_train)
        print("Clean Acc: {:.2f}".format(accuracy(y_pred1, y_test_all)))
        _, y_pred2 = model_fn(kernel_fn=kernel_fn, x_train=x_train_adv[-1], x_test=x_test_all, y_train=y_train_adv[-1])
        print("NTGA Robustness: {:.2f}".format(accuracy(y_pred2, y_test_all)))

    # Save poisoned data
    x_train_adv = jnp.concatenate(x_train_adv)[:x_train_all.shape[0]]   #get pData with the same size of the original
    y_train_adv = jnp.concatenate(y_train_adv)[:y_train_all.shape[0]]
    
    if args.dataset == "mnist":
        x_train_adv = x_train_adv.reshape(-1, 1, 28, 28)
    elif args.dataset == "cifar10":
        x_train_adv = x_train_adv.reshape(-1, 3, 32, 32)
    elif args.dataset == "imagenet":
        x_train_adv = x_train_adv.reshape(-1, 3, 224, 224)
    else:
        raise ValueError("Please specify the image size manually.")
    
    save_path = pjoin(args.save_path, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    x_train_adv = restore_pic(x_train_adv)
    jnp.save('{:s}/x_train_{:s}_ntga_{:s}_id-{:s}.npy'.format(save_path, args.dataset, args.fn_model_type, args.id), 
             x_train_adv)
    # jnp.save('{:s}/y_train_{:s}_ntga_{:s}_id-{:s}.npy'.format(save_path, args.dataset, args.fn_model_type, args.id),
    #           y_train_adv)
    jnp.save('{:s}/y_train_{:s}_ntga_{:s}_id-{:s}.npy'.format(save_path, args.dataset, args.fn_model_type, args.id),
              jnp.argmax(y_train_adv, axis=1))
    
    logger.log(logging.INFO, "================== Successfully generate NTGA! ==================")

def add_args(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices  #env must be set before torch.cude is called for the first time
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    args.device = device
    
    args_dataset = deepcopy(args)
    args_dataset.__dict__.clear()
    if args.fn_model_type == "fnn":
        args_dataset.flatten = True
    else:
        args_dataset.flatten = False
    # Epsilon, attack iteration, and step size
    if args.dataset == "mnist":
        args_dataset.num_classes = 10
        args_dataset.train_size = 60000 - args.val_size
        if args.eps == None:
            args_dataset.eps = 0.3
        else:
            args_dataset.eps = args.eps
    elif args.dataset == "cifar10":
        args_dataset.num_classes = 10
        args_dataset.train_size = 50000 - args.val_size
        if args.eps == None:
            args_dataset.eps = 8/255 
        else:
            args_dataset.eps = args.eps
    elif args.dataset == "imagenet":
        args_dataset.num_classes = 2
        args_dataset.train_size = 2220
        if args.eps == None:
            args_dataset.eps = 0.1
        else:
            args_dataset.eps = args.eps
        print("For ImageNet, please specify the file path manually.")
    else:
        raise ValueError("To load custom dataset, please modify the code directly.")
    args_dataset.use_entire_dataset = True
    args_dataset.eps_iter = (args_dataset.eps/args.nb_iter)* args.step_size  #!
    args_dataset.num_workers = 10
    return args_dataset, args


# if __name__ == '__main__':
current_time = datetime.datetime.now()
current_time = current_time.strftime("%m%d-%H_%M_%S")
with measure_time():
    args_general = get_args()
    args_dataset, args_general = add_args(args_general)
    args = combine_args(args_general, args_dataset)
    logger._init(pjoin(args.model_dir, args.save_name, f'train-{current_time}.log'))

    become_deterministic(args.seed)
    #torch.set_default_dtype(torch.float64)

    file_saver._init(args.save_name, args.model_dir, args)
    file_saver.save(save_type='setup', save_name='args:', value=args)

    logger.log(logging.INFO, f'\nsave_name: {args.save_name} \n', color='blue'.upper())
    main(args)

logger.log(logging.INFO, '--------------- Training finished ---------------', color='green'.upper())
logger.log(logging.INFO, f'model_name:{args.save_name}')    


# if __name__ == "__main__":
#     flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
#     flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
#     flags.DEFINE_bool(
#         "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
#     )

#     app.run(main)