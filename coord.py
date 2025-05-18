"""Tools for manipulating coordinate spaces and distances along rays."""

import jax
import jax.numpy as jnp
import torch


def parameterize_rays(H, W, focal, near, rays_o, rays_d, A, M):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2] * 0.5 * A
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2] * 0.5 * A
    o2 = 0.5 * M + (0.5 * M + M * near) / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2]) * 0.5 * A
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2]) * 0.5 * A
    d2 = (-M * near - 0.5 * M) / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def contract(x):
  """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
  eps = jnp.finfo(jnp.float32).eps
  # Clamping to eps prevents non-finite gradients when x == 0.
  x_mag_sq = jnp.maximum(eps, jnp.sum(x**2, axis=-1, keepdims=True))
  z = jnp.where(x_mag_sq <= 1, x, ((2 * jnp.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
  return z


def inv_contract(z):
  """The inverse of contract()."""
  eps = jnp.finfo(jnp.float32).eps
  # Clamping to eps prevents non-finite gradients when z == 0.
  z_mag_sq = jnp.maximum(eps, jnp.sum(z**2, axis=-1, keepdims=True))
  x = jnp.where(z_mag_sq <= 1, z, z / (2 * jnp.sqrt(z_mag_sq) - z_mag_sq))
  return x


def track_linearize(fn, mean, cov):
  """Apply function `fn` to a set of means and covariances, ala a Kalman filter.

  We can analytically transform a Gaussian parameterized by `mean` and `cov`
  with a function `fn` by linearizing `fn` around `mean`, and taking advantage
  of the fact that Covar[Ax + y] = A(Covar[x])A^T (see
  https://cs.nyu.edu/~roweis/notes/gaussid.pdf for details).

  Args:
    fn: the function applied to the Gaussians parameterized by (mean, cov).
    mean: a tensor of means, where the last axis is the dimension.
    cov: a tensor of covariances, where the last two axes are the dimensions.

  Returns:
    fn_mean: the transformed means.
    fn_cov: the transformed covariances.
  """
  if (len(mean.shape) + 1) != len(cov.shape):
    raise ValueError('cov must be non-diagonal')
  fn_mean, lin_fn = jax.linearize(fn, mean)
  fn_cov = jax.vmap(lin_fn, -1, -2)(jax.vmap(lin_fn, -1, -2)(cov))
  return fn_mean, fn_cov


def construct_ray_warps(fn, t_near, t_far):
  """Construct a bijection between metric distances and normalized distances.

  See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
  detailed explanation.

  Args:
    fn: the function to ray distances.
    t_near: a tensor of near-plane distances.
    t_far: a tensor of far-plane distances.

  Returns:
    t_to_s: a function that maps distances to normalized distances in [0, 1].
    s_to_t: the inverse of t_to_s.
  """
  if fn is None:
    fn_fwd = lambda x: x
    fn_inv = lambda x: x
  elif fn == 'piecewise':
    # Piecewise spacing combining identity and 1/x functions to allow t_near=0.
    fn_fwd = lambda x: jnp.where(x < 1, .5 * x, 1 - .5 / x)
    fn_inv = lambda x: jnp.where(x < .5, 2 * x, .5 / (1 - x))
  else:
    inv_mapping = {
        'reciprocal': jnp.reciprocal,
        'log': jnp.exp,
        'exp': jnp.log,
        'sqrt': jnp.square,
        'square': jnp.sqrt
    }
    fn_fwd = fn
    fn_inv = inv_mapping[fn.__name__]

  s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
  t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
  s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
  return t_to_s, s_to_t




def pos_enc(x, min_deg, max_deg, append_identity=True):
  """The positional encoding used by the original NeRF paper."""
  scales = 2**jnp.arange(min_deg, max_deg)
  shape = x.shape[:-1] + (-1,)
  scaled_x = jnp.reshape((x[..., None, :] * scales[:, None]), shape)
  # Note that we're not using safe_sin, unlike IPE.
  four_feat = jnp.sin(
      jnp.concatenate([scaled_x, scaled_x + 0.5 * jnp.pi], axis=-1))
  if append_identity:
    return jnp.concatenate([x] + [four_feat], axis=-1)
  else:
    return four_feat
