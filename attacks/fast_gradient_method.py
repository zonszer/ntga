# -*- coding: utf-8 -*-
import jax.numpy as np
from attacks.utils import one_hot
import jax

def fast_gradient_method(model_fn, kernel_fn, grads_fn, x_train, y_remain, x_remain, x_train_target, t=None, 
                         loss=None, fx_train_0=0., fx_test_0=0., eps=None, norm=None, 
                         clip_min=None, clip_max=None, targeted=False, batch_size=None,
                         T=None):
    """
    This code is based on CleverHans library(https://github.com/cleverhans-lab/cleverhans).
    JAX implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param kernel_fn: a callable that takes an input tensor and returns the kernel matrix.
    :param grads_fn: a callable that takes an input tensor and a loss function, 
            and returns the gradient w.r.t. an input tensor.
    :param x_train: input tensor (training data).
    :param x_test: input tensor (test data; used for evaluation).
    :param y_train: Tensor with one-hot true labels of training data.
    :param y_test: Tensor with one-hot true labels of test data. If targeted is true, then provide the
            target one-hot label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting poisoned data. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None. This argument does not have
            to be a binary one-hot label (e.g., [0, 1, 0, 0]), it can be floating points values
            that sum up to 1 (e.g., [0.05, 0.85, 0.05, 0.05]).
    :param t: time step used to compute poisoned data.
    :param loss: loss function.
    :param fx_train_0 = output of the network at `t == 0` on the training set. `fx_train_0=None`
            means to not compute predictions on the training set. fx_train_0=0. for infinite width.
    :param fx_test_0 = output of the network at `t == 0` on the test set. `fx_test_0=None`
            means to not compute predictions on the test set. fx_test_0=0. for infinite width.
            For more details, please refer to equations (10) and (11) in Wide Neural Networks of 
            Any Depth Evolve as Linear Models Under Gradient Descent (J. Lee and L. Xiao et al. 2019). 
            Paper link: https://arxiv.org/pdf/1902.06720.pdf.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
    :return: a tensor for the poisoned data.
    """
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
        
    x = x_train
    
    # if y_test is None: 
        # Using model predictions as ground truth to avoid label leaking
        # y_test = get_NTmodel_pred()
        ##maybe need to transform to values that all sum to 1 TODO:
        
    # Objective function - Θ(test, train)Θ(train, train)^-1(1-e^{-eta*t*Θ(train, train)})y_train
    if batch_size is None:
        batch_size = x_train.shape[0]   
    grads = 0

    # for i in range(int(len(x_test)/batch_size)):  #TODO: 是不是这里最好还是有cycle比较好（取所有测试集中所有标签相同的图的logits，拟合其分布？）
    # x_train = jax.device_put(x_train, jax.devices()[1])
    # x_remain = jax.device_put(x_remain, jax.devices()[1])
    # y_remain = jax.device_put(y_remain, jax.devices()[1])
    # x_train_target = jax.device_put(x_train_target, jax.devices()[1])
    batch_grads = grads_fn(x_train,
                            x_remain,     #   X_test.shape != X_train.shape
                            y_remain,
                            x_train_target,
                            kernel_fn,
                            loss,
                            t,   #!t is used to compute poisoned data
                            T) #主要还是不知道它grad ascent是咋实现？A: +grad 为ascent, -grad为descent
    grads += batch_grads    #!cumulative sum of all batches in test set


    axis = list(range(1, len(grads.shape)))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        perturbation = eps * np.sign(grads) #perturbation.shape = (512, 3072)
    elif norm == 1:
        raise NotImplementedError("L_1 norm has not been implemented yet.")
    elif norm == 2:
        square = np.maximum(avoid_zero_div, np.sum(np.square(grads), axis=axis, keepdims=True))
        perturbation = grads / np.sqrt(square)  #！
    else:
        raise ValueError("Norm order must be either np.inf or 2.")
    adv_x = x + perturbation    
    
    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = np.clip(adv_x, a_min=clip_min, a_max=clip_max)
        
    return adv_x
