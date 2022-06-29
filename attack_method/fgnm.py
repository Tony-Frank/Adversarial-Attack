"""The Fast Gradient Method attack."""

import numpy as np
import tensorflow as tf

from cleverhans.tf2.utils import optimize_linear, compute_gradient

def l2Sum(x):
    x = tf.reshape(x, [x.shape[0], -1])
    return tf.math.reduce_sum(x ** 2, -1)

def affine(x, maxValue: float):
    boost = l2Sum(x)
    zero_ = tf.zeros_like(boost) + 1e-6
    boost = tf.where(boost<1e-20, x=zero_, y=boost)
    scale = tf.sqrt(l2Sum(tf.sign(x)) / boost)
    # with tf.control_dependencies([tf.print(tf.reduce_max(scale), tf.reduce_min(scale))]):
        # x = tf.identity(x)
    scale = tf.expand_dims(scale, -1)
    scale = tf.expand_dims(scale, -1)
    scale = tf.expand_dims(scale, -1)
    if maxValue is not None:
        scale *= maxValue
    return scale * x

def fgnm(
    model_fn,
    x,
    eps,
    norm,
    loss_fn=None,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
):
    """
    Tensorflow 2.0 implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param loss_fn: (optional) callable. Loss function that takes (labels, logits) as arguments and returns loss.
                    default function is 'tf.nn.sparse_softmax_cross_entropy_with_logits'
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if loss_fn is None:
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min     is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    # cast to tensor if provided as numpy array
    x = tf.cast(x, tf.float32)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model_fn.predict(x), 1)

    grad = compute_gradient(model_fn, loss_fn, x, y, targeted)
    #optimal_perturbation = optimize_linear(grad, eps, norm)
    optimal_perturbation = affine(grad, None)
    
    # tf.multiply(eps, tf.multiply(tf.norm(tf.sign(grad)),tf.divide(grad, tf.norm(grad))))
    # Add perturbation to original example to obtain adversarial example
    adv_x = x + eps * optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    # print("norm********************************************************************** = {}".format(tf.norm(optimal_perturbation)))
    return adv_x
