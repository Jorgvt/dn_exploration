from typing import Any, Union, Callable, Sequence
import jax
from jax import random, numpy as jnp, lax
import flax.linen as nn
from einops import rearrange, reduce, repeat
from fxlayers.initializers import (
    bounded_uniform,
    displaced_normal,
    freq_scales_init,
    k_array,
    equal_to,
    linspace,
)
from fxlayers.layers import (
    pad_same_from_kernel_size,
    GDNGamma,
    GaussianLayerGamma,
    GDNGaussian,
)
from perceptualtests.color_matrices import Mng2xyz, Mxyz2atd

from .initialization import init_cs, init_dn_cs, init_v1, init_dn_v1
from .utils import rgb2atd

class CenterSurroundLogSigmaK(nn.Module):
    """Parametric center surround layer that optimizes log(sigma) instead of sigma and has a factor K instead of a second sigma."""
    features: int
    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    feature_group_count: int = 1
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    use_bias: bool = False
    xmean: float = 0.5
    ymean: float = 0.5
    fs: float = 1 # Sampling frequency
    normalize_prob: bool = True
    normalize_energy: bool = True
    normalize_sum: bool = True
    substraction_factor: float = 1.

    @nn.compact
    def __call__(self,
                 inputs,
                 train=False,
                 ):
        is_initialized = self.has_variable("precalc_filter", "kernel")
        precalc_filters = self.variable("precalc_filter",
                                        "kernel",
                                        jnp.zeros,
                                        (self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
        logsigma = self.param("logsigma",
                           bounded_uniform(minval=-2.2, maxval=-1.7),
                           (self.features*inputs.shape[-1],))
        K = self.param("K",
                           displaced_normal(mean=1.1, stddev=0.1),
                           (self.features*inputs.shape[-1],))
        A = self.param("A",
                       nn.initializers.ones,
                       (self.features*inputs.shape[-1],))
        sigma = jnp.exp(logsigma)
        sigma2 = K*sigma

        substraction_factor = jnp.zeros(shape=(self.features*inputs.shape[-1],))
        substraction_factor = substraction_factor.at[0].set(self.substraction_factor)

        if self.use_bias: bias = self.param("bias",
                                            self.bias_init,
                                            (self.features,))
        else: bias = 0.
        if is_initialized and not train: 
            kernel = precalc_filters.value
        elif is_initialized and train: 
            x, y = self.generate_dominion()
            kernel = jax.vmap(self.center_surround, in_axes=(None,None,None,None,0,0,0,None,None,None,0), out_axes=0)(x, y, self.xmean, self.ymean, sigma, sigma2, A, self.normalize_prob, self.normalize_energy, self.normalize_sum, substraction_factor)
            # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
            kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=self.features)
            precalc_filters.value = kernel
        else:
            kernel = precalc_filters.value

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4: inputs = inputs[None,:]; had_batch = False
        else: had_batch = True
        outputs = lax.conv(jnp.transpose(inputs,[0,3,1,2]),    # lhs = NCHW image tensor
               jnp.transpose(kernel,[3,2,0,1]), # rhs = OIHW conv kernel tensor
               (self.strides, self.strides),
               self.padding)
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0,2,3,1))
        if not had_batch: outputs = outputs[0]
        return outputs + bias

    # @staticmethod
    # def gaussian(x, y, xmean, ymean, sigma, A=1, normalize_prob=True):
    #     # A_norm = 1/(2*jnp.pi*sigma) if normalize_prob else 1.
    #     A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*sigma), 1.)
    #     return A*A_norm*jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2))
    
    @staticmethod
    def center_surround(x, y, xmean, ymean, sigma, sigma2, A=1, normalize_prob=True, normalize_energy=False, normalize_sum=False, substraction_factor=1.):
        def gaussian(x, y, xmean, ymean, sigma, A=1, normalize_prob=True, normalize_sum=False):
            A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*sigma**2), 1.)
            g = jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2))
            A_sum = jnp.where(normalize_sum, 1/g.sum(), 1.)
            return A*A_norm*A_sum*g
        g1 = gaussian(x, y, xmean, ymean, sigma, 1, normalize_prob, normalize_sum)
        g2 = gaussian(x, y, xmean, ymean, sigma2, 1, normalize_prob, normalize_sum)
        g = g1 - substraction_factor*g2
        E_norm = jnp.where(normalize_energy, jnp.sqrt(jnp.sum(g**2)), 1.)
        # A_sum = jnp.where(normalize_sum, g.sum(axis=(0,1), keepdims=True), 1.)
        # g = g/A_sum
        return A*g/E_norm
    
    # @staticmethod
    # def center_surround(x, y, xmean, ymean, sigma,  K, A=1, normalize_prob=True):
    #     return (1/(2*jnp.pi*sigma**2))*(jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2)) - (1/(K**2))*jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*(K*sigma)**2)))

    def return_kernel(self, params, c_in):
        x, y = self.generate_dominion()
        kernel = jax.vmap(self.center_surround, in_axes=(None,None,None,None,0,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, jnp.exp(params["params"]["logsigma"]), params["params"]["K"]*jnp.exp(params["params"]["logsigma"]), params["params"]["A"], self.normalize_prob, self.normalize_energy)
        # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, 3, self.features))
        kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=c_in, c_out=self.features)
        return kernel
    
    def generate_dominion(self):
        return jnp.meshgrid(jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size), jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size))


class DN(nn.Module):
    """---"""

    kernel_size: Sequence[int]
    apply_independently: bool = False
    inputs_star: Union[float, Sequence[float]] = 1.
    alpha: float = 2.
    epsilon: float = 1/2
    fs: int = 1
    normalize_prob: bool = False
    normalize_energy: bool = False
    normalize_sum: bool = True
    padding: str = "symmetric"
    use_noise: bool = False
    mean_lh: bool = False

    @nn.compact
    def __call__(self,
                 inputs,
                 train=False,
                 **kwargs,
                ):
        is_initialized_star = self.has_variable("batch_stats", "inputs_star")
        is_initialized_K = self.has_variable("batch_stats", "K")
        inputs_star = self.variable("batch_stats", "inputs_star", jnp.ones, (1,1,1,inputs.shape[-1]))
        K = self.variable("batch_stats", "K", jnp.ones, (1,1,1,inputs.shape[-1]))

        # if is_initialized_star and train:
        #     inputs_star.value = (inputs_star.value + jnp.quantile(jnp.abs(inputs), q=0.95, axis=(1,2), keepdims=True))/2

        # if is_initialized_K and train:
        #     K.value = (K.value + jnp.quantile(jnp.abs(inputs), q=0.95, axis=(1,2), keepdims=True))/2

        if self.use_noise:
            key = random.PRNGKey(42)
            a_star = inputs_star.value + (inputs_star.value/10)*random.normal(key, shape=inputs.shape)
            a_star = jnp.abs(a_star)
        else:
            a_star = inputs_star.value

        H = GDNGaussian(kernel_size=self.kernel_size, fs=self.fs, apply_independently=True, padding=self.padding,
                        normalize_prob=self.normalize_prob, normalize_energy=self.normalize_energy, normalize_sum=self.normalize_sum,
                        alpha=self.alpha, epsilon=self.epsilon,
                        )
        sign = jnp.sign(inputs)
        rh = H(jnp.abs(inputs), train=train)
        lh = H(jnp.broadcast_to(a_star, inputs.shape), train=train)
        if self.mean_lh:
            lh = jnp.mean(lh)
        return sign*(K.value/lh)*rh


class GaborLayerGammaHumanLike_(nn.Module):
    """Parametric Gabor layer with particular initialization."""

    n_scales: Sequence[int]  # [A, T, D]
    n_orientations: Sequence[int]  # [A, T, D]

    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    feature_group_count: int = 1

    use_bias: bool = False
    xmean: float = 0.5
    ymean: float = 0.5
    fs: float = 1  # Sampling frequency
    phase: Sequence[float] = (0., jnp.pi/2.)

    normalize_prob: bool = True
    normalize_energy: bool = False
    zero_mean: bool = False
    train_A: bool = False

    @nn.compact
    def __call__(
        self,
        inputs,
        train=False,
        return_freq=False,
        return_theta=False,
    ):
        total_scales = jnp.sum(jnp.array(self.n_scales))
        total_orientations = jnp.sum(jnp.array(self.n_orientations))
        phase = jnp.array(self.phase)
        features = jnp.sum(
            jnp.array(
                [
                    s * o * len(phase)
                    for s, o in zip(self.n_scales, self.n_orientations)
                ]
            )
        )

        is_initialized = self.has_variable("precalc_filter", "kernel")
        precalc_filters = self.variable(
            "precalc_filter",
            "kernel",
            jnp.zeros,
            (self.kernel_size, self.kernel_size, inputs.shape[-1], features+2), # To include the f0s
        )
        freq_a = self.param(
            "freq_a",
            freq_scales_init(n_scales=self.n_scales[0], fs=self.fs),
            (self.n_scales[0],),
        )
        gammax_a = self.param(
            "gammax_a", k_array(k=0.4, arr=1 / (freq_a**0.8)), (self.n_scales[0],)
        )
        gammay_a = self.param("gammay_a", equal_to(gammax_a * 0.8), (self.n_scales[0],))
        theta_a = self.param(
            "theta_a",
            linspace(start=0, stop=jnp.pi, num=self.n_orientations[0]),
            (self.n_orientations[0],),
        )
        sigma_theta_a = self.param(
            "sigma_theta_a", equal_to(theta_a), (self.n_orientations[0],)
        )

        freq_t = self.param(
            "freq_t",
            freq_scales_init(n_scales=self.n_scales[1], fs=self.fs),
            (self.n_scales[1],),
        )
        gammax_t = self.param(
            "gammax_t", k_array(k=0.4, arr=1 / (freq_t**0.8)), (self.n_scales[1],)
        )
        gammay_t = self.param("gammay_t", equal_to(gammax_t * 0.8), (self.n_scales[1],))
        # gamma_t_0 = self.param("gamma_t_0", nn.initializers.ones_init(), (1))
        theta_t = self.param(
            "theta_t",
            linspace(start=0, stop=jnp.pi, num=self.n_orientations[1]),
            (self.n_orientations[1],),
        )
        sigma_theta_t = self.param(
            "sigma_theta_t", equal_to(theta_t), (self.n_orientations[1],)
        )

        freq_d = self.param(
            "freq_d",
            freq_scales_init(n_scales=self.n_scales[2], fs=self.fs),
            (self.n_scales[2],),
        )
        gammax_d = self.param(
            "gammax_d", k_array(k=0.4, arr=1 / (freq_d**0.8)), (self.n_scales[2],)
        )
        gammay_d = self.param("gammay_d", equal_to(gammax_d * 0.8), (self.n_scales[2],))
        # gamma_d_0 = self.param("gamma_d_0", nn.initializers.ones_init(), (1,))
        theta_d = self.param(
            "theta_d",
            linspace(start=0, stop=jnp.pi, num=self.n_orientations[2]),
            (self.n_orientations[2],),
        )
        sigma_theta_d = self.param(
            "sigma_theta_d", equal_to(theta_d), (self.n_orientations[2],)
        )

        A = self.param("A", nn.initializers.ones_init(), (inputs.shape[-1], 128+2))
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (features,))
        else:
            bias = 0.0

        if is_initialized and not train:
            kernel = precalc_filters.value
        elif is_initialized and train:
            x, y = self.generate_dominion()
            ## A
            kernel_a = jax.vmap(
                self.gabor,
                in_axes=( None, None, None, None, 0, 0, 0, None, None, None, None, None, None, None,),
                out_axes=0,
            )
            kernel_a = jax.vmap(
                kernel_a,
                in_axes=( None, None, None, None, None, None, None, 0, 0, None, None, None, None, None,),
                out_axes=0,
            )
            kernel_a = jax.vmap(
                kernel_a,
                in_axes=( None, None, None, None, None, None, None, None, None, 0, None, None, None, None,),
                out_axes=0,
            )( x, y, self.xmean, self.ymean, gammax_a, gammay_a, freq_a, theta_a, sigma_theta_a, phase, 1, self.normalize_prob, self.normalize_energy, self.zero_mean,
            )
            kernel_a = rearrange(
                kernel_a, "phases rots fs_sigmas kx ky -> kx ky (phases rots fs_sigmas)"
            )
            kernel_a = repeat(
                kernel_a,
                "kx ky c_out -> kx ky c_in c_out",
                c_in=inputs.shape[-1],
                c_out=kernel_a.shape[-1],
            )

            ## T
            kernel_t = jax.vmap(
                self.gabor,
                in_axes=( None, None, None, None, 0, 0, 0, None, None, None, None, None, None, None,),
                out_axes=0,
            )
            kernel_t = jax.vmap(
                kernel_t,
                in_axes=( None, None, None, None, None, None, None, 0, 0, None, None, None, None, None,),
                out_axes=0,
            )
            kernel_t = jax.vmap(
                kernel_t,
                in_axes=( None, None, None, None, None, None, None, None, None, 0, None, None, None, None,),
                out_axes=0,
            )( x, y, self.xmean, self.ymean, gammax_t, gammay_t, freq_t, theta_t, sigma_theta_t, phase, 1, self.normalize_prob, self.normalize_energy, self.zero_mean,)
            kernel_t = rearrange(
                kernel_t, "phases rots fs_sigmas kx ky -> kx ky (phases rots fs_sigmas)"
            )
            kernel_t = repeat(
                kernel_t,
                "kx ky c_out -> kx ky c_in c_out",
                c_in=inputs.shape[-1],
                c_out=kernel_t.shape[-1],
            )
            ## Generate the f=0 residual
            gabor_t_0 = self.gabor(x, y, self.xmean, self.ymean, 1/0.4, 1/0.4, 0., 0., 0., 0., 1., self.normalize_prob, self.normalize_energy, False)
            gabor_t_0 = repeat(
                gabor_t_0,
                "kx ky -> kx ky c_in c_out",
                c_in=inputs.shape[-1],
                c_out=1,
            )
            kernel_t = jnp.concatenate(
                [gabor_t_0, kernel_t], axis=-1,
            )

            ## D
            kernel_d = jax.vmap(
                self.gabor,
                in_axes=( None, None, None, None, 0, 0, 0, None, None, None, None, None, None, None,),
                out_axes=0,
            )
            kernel_d = jax.vmap(
                kernel_d,
                in_axes=( None, None, None, None, None, None, None, 0, 0, None, None, None, None, None,),
                out_axes=0,
            )
            kernel_d = jax.vmap(
                kernel_d,
                in_axes=( None, None, None, None, None, None, None, None, None, 0, None, None, None, None,),
                out_axes=0,
            )( x, y, self.xmean, self.ymean, gammax_d, gammay_d, freq_d, theta_d, sigma_theta_d, phase, 1, self.normalize_prob, self.normalize_energy, self.zero_mean,)
            kernel_d = rearrange(
                kernel_d, "phases rots fs_sigmas kx ky -> kx ky (phases rots fs_sigmas)"
            )
            kernel_d = repeat(
                kernel_d,
                "kx ky c_out -> kx ky c_in c_out",
                c_in=inputs.shape[-1],
                c_out=kernel_d.shape[-1],
            )

            ## Generate the f=0 residual
            gabor_d_0 = self.gabor(x, y, self.xmean, self.ymean, 1/0.4, 1/0.4, 0., 0., 0., 0., 1., self.normalize_prob, self.normalize_energy, False)
            gabor_d_0 = repeat(
                gabor_d_0,
                "kx ky -> kx ky c_in c_out",
                c_in=inputs.shape[-1],
                c_out=1,
            )
            kernel_d = jnp.concatenate(
                [gabor_d_0, kernel_d], axis=-1,
            )
            ## Concat all of them
            kernel = jnp.concatenate([kernel_a, kernel_t, kernel_d], axis=-1)
            kernel = kernel * A[None, None, :, :]
            precalc_filters.value = kernel
        else:
            kernel = precalc_filters.value

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4:
            inputs = inputs[None, :]
            had_batch = False
        else:
            had_batch = True
        outputs = lax.conv(
            jnp.transpose(inputs, [0, 3, 1, 2]),  # lhs = NCHW image tensor
            jnp.transpose(kernel, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
            (self.strides, self.strides),
            self.padding,
        )
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0, 2, 3, 1))
        fmean = jnp.concatenate(
            [
                jnp.tile(jnp.tile(f, reps=len(t)), reps=len(self.phase))
                for f, t in zip([freq_a, freq_t, freq_d], [theta_a, theta_t, theta_d])
            ]
        )
        ## Include the f0s
        fmean = jnp.insert(fmean, jnp.array([64, 64+32]), jnp.array([0., 0.]))

        theta_mean = jnp.concatenate(
            [
                jnp.tile(jnp.repeat(t, repeats=len(f)), reps=len(self.phase))
                for f, t in zip([freq_a, freq_t, freq_d], [theta_a, theta_t, theta_d])
            ]
        )
        theta_mean = jnp.insert(theta_mean, jnp.array([64, 64+32]), jnp.array([0., 0.]))

        if not had_batch:
            outputs = outputs[0]
        if return_freq and return_theta:
            return outputs + bias, fmean, theta_mean
        elif return_freq and not return_theta:
            return outputs + bias, fmean
        elif not return_freq and return_theta:
            return outputs + bias, theta_mean
        else:
            return outputs + bias

    @staticmethod
    def gabor(
        x,
        y,
        xmean,
        ymean,
        gammax,
        gammay,
        freq,
        theta,
        sigma_theta,
        phase,
        A=1,
        normalize_prob=True,
        normalize_energy=False,
        zero_mean=False,
    ):
        x, y = x - xmean, y - ymean
        ## Obtain the normalization coeficient
        gamma_vector = jnp.array([gammax, gammay])
        # jax.debug.print(f"Freq: {freq}")
        # jax.debug.print(f"Gamma Vector: {gamma_vector.shape}")
        # jax.debug.print(f"Gamma Vector: {gamma_vector}")
        inv_cov_matrix = jnp.diag(gamma_vector) ** 2
        # jax.debug.print(f"Inv Cov Matrix: {inv_cov_matrix.shape}")
        # det_cov_matrix = 1/jnp.linalg.det(cov_matrix)
        # # A_norm = 1/(2*jnp.pi*jnp.sqrt(det_cov_matrix)) if normalize_prob else 1.
        # A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*jnp.sqrt(det_cov_matrix)), 1.)
        A_norm = 1.0

        ## Rotate the sinusoid
        rotation_matrix = jnp.array(
            [
                [jnp.cos(sigma_theta), -jnp.sin(sigma_theta)],
                [jnp.sin(sigma_theta), jnp.cos(sigma_theta)],
            ]
        )
        # jax.debug.print(f"Rotation Matrix: {rotation_matrix.shape}")
        rotated_covariance = (
            rotation_matrix @ inv_cov_matrix @ jnp.transpose(rotation_matrix)
        )
        x_r_1 = rotated_covariance[0, 0] * x + rotated_covariance[0, 1] * y
        y_r_1 = rotated_covariance[1, 0] * x + rotated_covariance[1, 1] * y
        distance = x * x_r_1 + y * y_r_1
        g = (
            A_norm
            * jnp.exp(-distance / 2)
            * jnp.cos(
                2 * jnp.pi * freq * (x * jnp.cos(theta) + y * jnp.sin(theta)) + phase
            )
        )
        g = jnp.where(zero_mean, g - g.mean(), g)
        E_norm = jnp.where(normalize_energy, jnp.sqrt(jnp.sum(g**2)), 1.0)
        return A * g / E_norm

    def return_kernel(self, params, c_in=3):
        x, y = self.generate_dominion()
        sigmax, sigmay = jnp.exp(params["sigmax"]), jnp.exp(params["sigmay"])
        kernel = jax.vmap(
            self.gabor,
            in_axes=( None, None, None, None, 0, 0, None, None, None, None, None, None, None,),
            out_axes=0,
        )
        kernel = jax.vmap(
            kernel,
            in_axes=( None, None, None, None, None, None, 0, None, None, None, None, None, None,),
            out_axes=0,
        )
        kernel = jax.vmap(
            kernel,
            in_axes=( None, None, None, None, None, None, None, 0, 0, 0, None, None, None,),
            out_axes=0,
        )(
            x,
            y,
            self.xmean,
            self.ymean,
            params["sigmax"],
            params["sigmay"],
            params["freq"],
            params["theta"],
            params["sigma_theta"],
            phase,
            1,
            self.normalize_prob,
            self.normalize_energy,
        )
        # kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=self.features)
        kernel = rearrange(kernel, "rots fs sigmas kx ky -> kx ky (rots fs sigmas)")
        kernel = repeat(
            kernel, "kx ky c_out -> kx ky c_in c_out", c_in=c_in, c_out=kernel.shape[-1]
        )
        return kernel

    def generate_dominion(self):
        return jnp.meshgrid(
            jnp.linspace(0, self.kernel_size / self.fs, num=self.kernel_size),
            jnp.linspace(0, self.kernel_size / self.fs, num=self.kernel_size),
        )


class GDNControl(nn.Module):
    """---"""

    kernel_size: Sequence[int]
    apply_independently: bool = False
    inputs_star: Union[float, Sequence[float]] = 1.
    alpha: float = 2.
    epsilon: float = 1/2
    fs: int = 1
    normalize_prob: bool = False
    normalize_energy: bool = False
    normalize_sum: bool = True
    # K: Sequence[float] = 1.

    @nn.compact
    def __call__(self,
                 inputs,
                 fmean,
                 theta_mean,
                 train=False,
                 **kwargs,
                ):
        is_initialized_star = self.has_variable("batch_stats", "inputs_star")
        is_initialized_K = self.has_variable("batch_stats", "K")
        inputs_star = self.variable("batch_stats", "inputs_star", jnp.ones, (1,1,1,inputs.shape[-1]))
        K = self.variable("batch_stats", "K", jnp.ones, (1,1,1,inputs.shape[-1]))
    
        # if is_initialized_star and train:
        #     inputs_star.value = (inputs_star.value + jnp.quantile(jnp.abs(inputs), q=0.95, axis=(1,2), keepdims=True))/2

        # if is_initialized_K and train:
        #     K.value = (K.value + jnp.quantile(jnp.abs(inputs), q=0.95, axis=(1,2), keepdims=True))/2

        H = GDNSpatioChromaFreqOrient(self.kernel_size, fs=self.fs, apply_independently=self.apply_independently, padding="symmetric",
                                      normalize_prob=self.normalize_prob, normalize_energy=self.normalize_energy, normalize_sum=self.normalize_sum,
                                      alpha=self.alpha, epsilon=self.epsilon)
        # inputs_star = jnp.abs(inputs).mean(axis=(0,1,2), keepdims=True)
        rh = H(inputs, fmean, theta_mean, train=train)
        lh = H(jnp.broadcast_to(inputs_star.value, inputs.shape), fmean, theta_mean, train=train)
        return (K.value/lh)*rh


class GDNSpatioChromaFreqOrient(nn.Module):
    """Generalized Divisive Normalization."""

    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    # inputs_star: float = 1.
    # outputs_star: Union[None, float] = None
    fs: int = 1
    apply_independently: bool = False
    bias_init: Callable = nn.initializers.ones_init()
    alpha: float = 2.0
    epsilon: float = 1 / 2  # Exponential of the denominator
    eps: float = 1e-6  # Numerical stability in the denominator
    normalize_prob: bool = False
    normalize_energy: bool = True
    normalize_sum: bool = False

    @nn.compact
    def __call__(
        self,
        inputs,
        fmean,
        theta_mean,
        train=False,
    ):
        b, h, w, c = inputs.shape
        bias = self.param(
            "bias",
            # equal_to(inputs_star/10),
            self.bias_init,
            (c,),
        )
        # is_initialized = self.has_variable("batch_stats", "inputs_star")
        # inputs_star = self.variable("batch_stats", "inputs_star", lambda x: jnp.ones(x)*self.inputs_star, (len(self.inputs_star),))
        # inputs_star_ = jnp.ones_like(inputs)*inputs_star.value
        GL = GaussianLayerGamma(
            features=c,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="VALID",
            fs=self.fs,
            xmean=self.kernel_size / self.fs / 2,
            ymean=self.kernel_size / self.fs / 2,
            normalize_prob=self.normalize_prob,
            normalize_energy=self.normalize_energy,
            normalize_sum=self.normalize_sum,
            use_bias=False,
            feature_group_count=c,
        )
        FOG = ChromaFreqOrientGaussianGamma(normalize_sum=self.normalize_sum)
        outputs = GL(
            pad_same_from_kernel_size(
                inputs, kernel_size=self.kernel_size, mode=self.padding
            )
            ** self.alpha,
            train=train,
        )  # /(self.kernel_size**2)
        outputs = FOG(outputs, fmean=fmean, theta_mean=theta_mean, train=train)

        ## Coef
        # coef = GL(inputs_star_**self.alpha, train=train)#/(self.kernel_size**2)
        # coef = FG(coef, fmean=fmean)
        # coef = rearrange(coef, "b h w (phase theta f) -> b h w (phase f theta)", b=b, h=h, w=w, phase=2, f=config.N_SCALES, theta=config.N_ORIENTATIONS)
        # coef = OG(coef, theta_mean=theta_mean) + bias
        # coef = rearrange(coef, "b h w (phase f theta) -> b h w (phase theta f)", b=b, h=h, w=w, phase=2, f=config.N_SCALES, theta=config.N_ORIENTATIONS)
        # coef = jnp.clip(coef+bias, a_min=1e-5)**self.epsilon
        # # coef = inputs_star.value * coef
        # if self.outputs_star is not None: coef = coef/inputs_star.value*self.outputs_star

        # if is_initialized and train:
        #     inputs_star.value = (inputs_star.value + jnp.quantile(jnp.abs(inputs), q=0.95, axis=(0,1,2)))/2
        # return coef * inputs / (jnp.clip(denom+bias, a_min=1e-5)**self.epsilon + self.eps)
        return (
            inputs
            / (jnp.clip(outputs + bias, a_min=1e-5) ** self.epsilon + self.eps)
        )


class ChromaFreqOrientGaussianGamma(nn.Module):
    """(1D) Gaussian interaction between gamma_fuencies and orientations optimizing gamma = 1/sigma instead of sigma."""

    use_bias: bool = False
    strides: int = 1
    padding: str = "SAME"
    bias_init: Callable = nn.initializers.zeros_init()
    n_scales: Sequence[int] = (4, 2, 2)
    n_orientations: Sequence[int] = (8, 8, 8)
    normalize_sum: bool = False

    @nn.compact
    def __call__(
        self,
        inputs,
        fmean,
        theta_mean,
        train=False,
        **kwargs,
    ):
        is_initialized = self.has_variable("precalc_filter", "kernel")
        precalc_filters = self.variable("precalc_filter",
                                        "kernel",
                                        jnp.zeros,
                                        (1,1,len(fmean), len(theta_mean)))
        gamma_f_a = self.param(
            "gamma_f_a",
            k_array(1 / 0.4, arr=jnp.array([2.0, 4.0, 8.0, 16.0])),
            (self.n_scales[0],),
        )
        gamma_theta_a = self.param(
            "gamma_theta_a",
            nn.initializers.ones_init(),
            #  (self.n_orientations[0],))
            (8,),
        )

        gamma_f_t = self.param(
            "gamma_f_t",
            k_array(1 / 0.4, arr=jnp.array([3.0, 6.0])),
            (self.n_scales[1],),
        )
        gamma_theta_t = self.param(
            "gamma_theta_t",
            nn.initializers.ones_init(),
            #  (self.n_orientations[1],))
            (8,),
        )

        gamma_f_d = self.param(
            "gamma_f_d",
            k_array(1 / 0.4, arr=jnp.array([3.0, 6.0])),
            (self.n_scales[2],),
        )
        gamma_theta_d = self.param(
            "gamma_theta_d",
            nn.initializers.ones_init(),
            #  (self.n_orientations[2],))
            (8,),
        )

        H_cc = self.param("H_cc", nn.initializers.ones_init(), (3, 3))

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (len(fmean),))
        else:
            bias = 0.0
        # n_groups = inputs.shape[-1] // len(fmean)

        if is_initialized and not train: 
            kernel = precalc_filters.value
        elif is_initialized and train: 

            ## Repeat gammas

            gamma_f = jnp.concatenate(
                [
                    jnp.tile(jnp.tile(f, reps=len(t)), reps=2)
                    for f, t in zip(
                        [gamma_f_a, gamma_f_t, gamma_f_d],
                        [gamma_theta_a, gamma_theta_t, gamma_theta_d],
                    )
                ]
            )
            gamma_f = jnp.insert(gamma_f, jnp.array([64, 64+32]), 1/(jnp.array([0.25, 0.25])/jnp.sqrt(2)))

            gamma_theta = jnp.concatenate(
                [
                    jnp.tile(jnp.repeat(t, repeats=len(f)), reps=2)
                    for f, t in zip(
                        [gamma_f_a, gamma_f_t, gamma_f_d],
                        [gamma_theta_a, gamma_theta_t, gamma_theta_d],
                    )
                ]
            )
            ## Include the f0s
            gamma_theta = jnp.insert(gamma_theta, jnp.array([64, 64+32]), 1/(jnp.pi/180*jnp.array([15., 15.])/jnp.sqrt(2)))

            ## Repeating
            cc = jnp.array([0, 1, 2])
            cc = jnp.repeat(
                cc, repeats=jnp.array([64, 32, 32]), total_repeat_length=len(fmean)
            )
            ## Phase index
            pp = jnp.array([0,1])
            pp = jnp.concatenate([
                jnp.repeat(pp, repeats=jnp.array([32,32]), total_repeat_length=64),
                jnp.repeat(pp, repeats=jnp.array([16,16]), total_repeat_length=32),
                jnp.repeat(pp, repeats=jnp.array([16,16]), total_repeat_length=32),
            ])
            pp = jnp.insert(pp, jnp.array([64, 64+32]), jnp.array([1., 1.]))
            H_pp = jnp.eye(2)

            kernel = jax.vmap(
                self.gaussian,
                in_axes=(None, None, 0, 0, 0, 0, None, 0, None, None, 0, None, None, None),
                out_axes=1,
            )(fmean, theta_mean, fmean, theta_mean, gamma_f, gamma_theta, cc, cc, H_cc, pp, pp, H_pp, 1, self.normalize_sum)
            kernel = kernel[None, None, :, :]
            A_sum = jnp.where(self.normalize_sum, 1/kernel.sum(axis=2, keepdims=True), 1.)
            kernel = A_sum*kernel
            precalc_filters.value = kernel
        else:
            kernel = precalc_filters.value

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4:
            inputs = inputs[None, :]
            had_batch = False
        else:
            had_batch = True
        outputs = lax.conv_general_dilated(
            jnp.transpose(inputs, [0, 3, 1, 2]),  # lhs = NCHW image tensor
            jnp.transpose(kernel, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
            (self.strides, self.strides),
            self.padding,
        )
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0, 2, 3, 1))
        if not had_batch:
            outputs = outputs[0]
        return outputs + bias

    @staticmethod
    def gaussian(
        f, theta, fmean, theta_mean, gamma_f, gamma_theta, c_1, c_2, H_cc, p_1, p_2, H_pp, A=1, norm_sum=False
    ):
        def diff_ang(ang1, ang2):
            return jnp.min(
                jnp.array([
                jnp.abs(ang1-ang2),
                jnp.abs(ang1+jnp.pi-ang2),
                jnp.abs(ang1-ang2-jnp.pi),
                jnp.abs(ang1+2*jnp.pi-ang2),
                jnp.abs(ang1-ang2-2*jnp.pi),
                ]), axis=0
            )
        g_f = jnp.exp(-((gamma_f**2) * (f - fmean) ** 2) / (2))
        g_theta =  jnp.exp(-((gamma_theta**2) * diff_ang(theta, theta_mean) ** 2) / (2))
        # A_f = jnp.where(norm_sum, 1/g_f.sum(), 1.)
        # A_theta = jnp.where(norm_sum, 1/g_theta.sum(), 1.)
        A_f = 1.
        A_theta = 1.

        return (
            H_cc[c_1, c_2]
            * H_pp[p_1, p_2]
            * A
            * A_f * g_f
            * A_theta * g_theta
        )


def ng2atd(img):
    return img @ Mng2xyz.T @ Mxyz2atd.T


class Model(nn.Module):

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs):

        outputs = GDNGamma()(inputs)
        outputs = ng2atd(outputs)

        outputs = pad_same_from_kernel_size(outputs, kernel_size=63, mode="symmetric")
        outputs = CenterSurroundLogSigmaK(
                            # features=3, kernel_size=31, fs=32,
                            # xmean=(31/32)/2,
                            # ymean=(31/32)/2,
                            features=3, kernel_size=64, fs=128,
                            xmean=(63/128)/2,
                            ymean=(63/128)/2,
                            normalize_prob=False,
                            normalize_energy=False,
                            normalize_sum=True,
                            substraction_factor=0.95,
                            padding="VALID")(outputs, **kwargs)

        outputs = DN(kernel_size=31, fs=31, apply_independently=True, normalize_energy=False, normalize_prob=False, normalize_sum=True)(outputs, **kwargs)

        outputs = nn.max_pool(outputs, window_shape=(4,4), strides=(4,4))

        outputs = pad_same_from_kernel_size(outputs, kernel_size=31, mode="symmetric")
        outputs, fmean, theta_mean = GaborLayerGammaHumanLike_(
            n_scales=[4, 2, 2],
            n_orientations=[8, 8, 8],
            kernel_size=31,
            fs=32,
            xmean=32 / 32 / 2,
            ymean=32 / 32 / 2,
            strides=1,
            padding="VALID",
            normalize_prob=False,
            normalize_energy=True,
            zero_mean=True,
            use_bias=False,
            train_A=False,
        )(outputs, return_freq=True, return_theta=True, **kwargs)

        outputs = GDNControl(kernel_size=31, fs=32, normalize_prob=False, normalize_energy=False, normalize_sum=True)(outputs, fmean, theta_mean, **kwargs)

        return outputs

class ModelCls(nn.Module):
    config: Any

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs):
        outputs = Model(name="perceptnet")(inputs, **kwargs)
        # outputs = reduce(outputs, "b h w c -> b c", reduction="mean")
        if self.config.GAP:
            outputs = outputs.mean(axis=(1,2))
        else:
            outputs = rearrange(outputs, "b h w c -> b (h w c)")
        outputs = nn.Dense(features=10)(outputs)
        return outputs

class SimpleTest(nn.Module):
    config: Any

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs):
        outputs = nn.Conv(features=32, kernel_size=(3,3))(inputs)
        outputs = nn.relu(outputs)
        outputs = nn.max_pool(outputs, (2,2), (2,2))

        outputs = nn.Conv(features=64, kernel_size=(3,3))(outputs)
        outputs = nn.relu(outputs)
        outputs = nn.max_pool(outputs, (2,2), (2,2))

        outputs = nn.Conv(features=128, kernel_size=(3,3))(outputs)
        outputs = nn.relu(outputs)
        outputs = nn.max_pool(outputs, (2,2), (2,2))

        outputs = rearrange(outputs, "b h w c -> b (h w c)")
        outputs = nn.Dense(10)(outputs)
        return outputs
