from jax import numpy as jnp

def init_dn_gamma(params):
    params_ = params.copy()
    params_["GDNGamma_0"]["bias"] = jnp.ones_like(params_["GDNGamma_0"]["bias"]) * 0.1
    params_["GDNGamma_0"]["kernel"] = (
        jnp.ones_like(params_["GDNGamma_0"]["kernel"]) * 0.5
    )
    # params_["GDNGamma_0"]["bias"] = jnp.ones_like(params_["GDNGamma_0"]["bias"]) * (-0.04)
    # params_["GDNGamma_0"]["kernel"] = (
    #     jnp.ones_like(params_["GDNGamma_0"]["kernel"]) * 0.216
    # )
    return params

## Center Surround
def init_cs(params):
    ## A
    s1 = 0.035/jnp.sqrt(2)
    s2 = 0.18/jnp.sqrt(2)
    ## T
    s1_t = 0.05/jnp.sqrt(2)
    s2_t = s1_t*10/jnp.sqrt(2)
    ## D
    s1_d = 0.07/jnp.sqrt(2)
    s2_d = s1_t*10/jnp.sqrt(2)
    params_ = params.copy()
    params_["logsigma"] = jnp.array(
        [jnp.log(s1), jnp.log(s1), jnp.log(s1),
        jnp.log(s1_t), jnp.log(s1_t), jnp.log(s1_t),
        jnp.log(s1_d), jnp.log(s1_d), jnp.log(s1_d)]
    )
    params_["K"] = jnp.array(
        [s2/s1, s2/s1, s2/s1,
        s2_t/s1_t, s2_t/s1_t, s2_t/s1_t,
        s2_d/s1_d, s2_d/s1_d, s2_d/s1_d]
    )
    params_["A"] = jnp.array(
        [1.0, 0.0, 0.0,
        0.0, 0.8, 0.0,
        0.0, 0.0, 0.6]
    )
    return params_

## DN - CS
def init_dn_cs(params, state):
    """
    K = cs_q
    a_star = cs_q
    use_noise = False
    sigma = 0.1
    mean_lh = False
    b = (cs_q**2)/10
    """
    params_ = params.copy()
    state_ = state.copy()
    a_star_cs = jnp.load("a_star_gdn_cs.npy")
    params_["DN_0"]["GDNGaussian_0"]["GaussianLayerGamma_0"]["gamma"] = (1/(0.1/jnp.sqrt(2)))*jnp.ones_like(params["DN_0"]["GDNGaussian_0"]["GaussianLayerGamma_0"]["gamma"])
    params_["DN_0"]["GDNGaussian_0"]["GaussianLayerGamma_0"]["bias"] = (a_star_cs.squeeze()**2)/10
    state_["batch_stats"]["DN_0"]["K"] = a_star_cs
    state_["batch_stats"]["DN_0"]["inputs_star"] = a_star_cs
    return params, state


def init_v1(params):
    params_ = params.copy()
    params_["GaborLayerGammaHumanLike__0"]["freq_a"] = jnp.array([2.0, 4., 8., 16.])
    params_["GaborLayerGammaHumanLike__0"]["freq_t"] = jnp.array([2., 4.])
    params_["GaborLayerGammaHumanLike__0"]["freq_d"] = jnp.array([2., 4.])

    A_a = jnp.zeros(shape=(3, 64), dtype=jnp.float32)
    A_a = A_a.at[0, :].set(1.0)
    A_t = jnp.zeros(shape=(3, 33), dtype=jnp.float32) # Add 1 to account for the f=0
    A_t = A_t.at[1, :].set(1.0)
    A_t = A_t.at[1,0].set(5.0)
    A_t = A_t/(2*1.2)
    A_d = jnp.zeros(shape=(3, 33), dtype=jnp.float32) # Add 1 to account for the f=0
    A_d = A_d.at[2, :].set(1.0)
    A_d = A_d.at[2,0].set(4.0)
    A_d = A_d/(1.5*1.2)
    params_["GaborLayerGammaHumanLike__0"]["A"] = jnp.concatenate(
        [A_a, A_t, A_d], axis=-1
    )
    params_["GaborLayerGammaHumanLike__0"]["gammax_a"] = 1/jnp.array([0.16, 0.08, 0.06, 0.04])
    params_["GaborLayerGammaHumanLike__0"]["gammay_a"] = 1/jnp.array([0.16, 0.08, 0.06, 0.04])
    return params_
def init_dn_v1(params, state):
    params_ = params.copy()
    state_ = state.copy()

    params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["H_cc"] = jnp.eye(3,3)


    params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_f_a"] = 1/(0.25*jnp.array([1., 1., 1., 1.])/jnp.sqrt(2))
    params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_f_t"] = 1/(0.25*jnp.array([1., 1.,])/jnp.sqrt(2))
    params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_f_d"] = 1/(0.25*jnp.array([1., 1.,])/jnp.sqrt(2))

    params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_a"] = 1/(jnp.pi/180*jnp.array([15.]*8)/jnp.sqrt(2))
    params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_t"] = 1/(jnp.pi/180*jnp.array([15.]*8)/jnp.sqrt(2))
    params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_d"] = 1/(jnp.pi/180*jnp.array([15.]*8)/jnp.sqrt(2))


    params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"] = jnp.concatenate([
        jnp.tile(jnp.tile((1/2)*params_["GaborLayerGammaHumanLike__0"]["gammax_a"], reps=8), reps=2),
        jnp.tile(jnp.tile((1/2)*params_["GaborLayerGammaHumanLike__0"]["gammax_t"], reps=8), reps=2),
        jnp.tile(jnp.tile((1/2)*params_["GaborLayerGammaHumanLike__0"]["gammax_d"], reps=8), reps=2),
    ])
    params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"] = jnp.insert(params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"],
                                                                                                         jnp.array([64, 64+32]),
                                                                                                         jnp.array([
                                                                                                            params_["GaborLayerGammaHumanLike__0"]["gammax_t"][0],
                                                                                                            params_["GaborLayerGammaHumanLike__0"]["gammax_d"][0],
                                                                                                         ]))

    inputs_star = jnp.load("a_star_gdn_v1.npy")
    params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["bias"] = (inputs_star.squeeze()**2)/1000
    # params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["bias"] = params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["bias"]**(1/3)

    state_["batch_stats"]["GDNControl_0"]["inputs_star"] = inputs_star

    state_["batch_stats"]["GDNControl_0"]["K"] = inputs_star
    coef = 2
    Wr = jnp.concatenate([
        jnp.tile(
            jnp.tile(
                jnp.array([1/(coef**3), 1/(coef**2), 1/coef, 1]),
                reps=8
            )
            , reps=2,
        ),
        jnp.tile(
            jnp.tile(
                jnp.array([1/(coef**3), 1/(coef**2)]),
                reps=8
            )
            , reps=2,
        ),
        jnp.tile(
            jnp.tile(
                jnp.array([1/(coef**3), 1/(coef**2)]),
                reps=8
            )
            , reps=2,
        )
    ])
    Wr = jnp.insert(Wr, jnp.array([64, 64+32]), jnp.array([1/coef**3, 1/coef**3]))
    state_["batch_stats"]["GDNControl_0"]["K"] = state_["batch_stats"]["GDNControl_0"]["K"]*Wr
    return params, state