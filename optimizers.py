import tensorflow as tf


class pSGLDOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, sigma=0.01, alpha=0.99, diagonal_bias=1e-5, name="SGOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self.alpha = alpha
        self.diagonal_bias = diagonal_bias
        self.sigma = sigma
#
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "rms")
#
    def _decayed_sigma(self, var_dtype):
        if isinstance(self.sigma, tf.keras.optimizers.schedules.LearningRateSchedule):
            local_step = tf.cast(self.iterations, var_dtype)
            sigma = tf.cast(self.sigma(local_step), var_dtype)
            return sigma
        else:
            return self.sigma
#
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        sigma = self._decayed_sigma(var_dtype)
        rms_var = self.get_slot(var, "rms")
        new_rms = self.alpha*rms_var + (1-self.alpha)*tf.square(grad)
        preconditioner = 1./(self.diagonal_bias + tf.math.sqrt(new_rms))
        stddev = sigma*tf.math.sqrt(lr_t*preconditioner)
        new_var = var - lr_t*preconditioner*grad + tf.random.normal(shape=tf.shape(grad), stddev=stddev)
        rms_var.assign(new_rms)
        var.assign(new_var)



class aSGLDOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, sigma=0.01, beta_1=0.9, beta_2=0.999, diagonal_bias=1e-5, name="SGOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.diagonal_bias = diagonal_bias
        self.sigma = sigma
#
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var,"m")
            self.add_slot(var,"v")
#
    def _decayed_sigma(self, var_dtype):
        if isinstance(self.sigma, tf.keras.optimizers.schedules.LearningRateSchedule):
            local_step = tf.cast(self.iterations, var_dtype)
            sigma = tf.cast(self.sigma(local_step), var_dtype)
            return sigma
        else:
            return self.sigma
#
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        sigma = self._decayed_sigma(var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        m_var = self.get_slot(var, "m")
        v_var = self.get_slot(var, "v")
        new_m = self.beta_1*m_var + (1.-self.beta_1)*grad
        new_v = self.beta_2*v_var + (1.-self.beta_2)*tf.square(grad)
        v_hat = new_v/(1.-tf.pow(self.beta_2, local_step))
        preconditioner =  1./ ((1.-tf.pow(self.beta_1, local_step)) * (self.diagonal_bias + tf.math.sqrt(v_hat)))
        stddev = sigma*tf.math.sqrt(lr_t*preconditioner)
        new_var = var - lr_t*preconditioner*grad + tf.random.normal(shape=tf.shape(grad),stddev=stddev)
        m_var.assign(new_m)
        v_var.assign(new_v)
        var.assign(new_var)



class pAdaSGLDOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, sigma=0.01, beta_1=0.9, beta_2=0.9, diagonal_bias=1e-5, name="AdadeltaOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.diagonal_bias = diagonal_bias
        self.sigma = sigma
#
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "delta_rms")
            self.add_slot(var, "grad_rms")
#
    def _decayed_sigma(self, var_dtype):
        if isinstance(self.sigma, tf.keras.optimizers.schedules.LearningRateSchedule):
            local_step = tf.cast(self.iterations, var_dtype)
            sigma = tf.cast(self.sigma(local_step), var_dtype)
            return sigma
        else:
            return self.sigma
#
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        sigma = self._decayed_sigma(var_dtype)
        delta_rms_var = self.get_slot(var, "delta_rms")
        grad_rms_var = self.get_slot(var, "grad_rms")
        new_grad_rms = self.beta_2*grad_rms_var + (1-self.beta_2)*tf.square(grad)
        preconditioner = tf.sqrt(delta_rms_var + tf.cast(self.diagonal_bias, grad.dtype))*tf.math.rsqrt(new_grad_rms + tf.cast(self.diagonal_bias, grad.dtype))
        new_delta = - lr_t*preconditioner*grad + tf.random.normal(shape=tf.shape(grad),stddev=tf.math.sqrt(lr_t)*tf.sqrt(preconditioner)*sigma)
        new_var = var + new_delta
        new_delta_rms = self.beta_1*delta_rms_var + (1-self.beta_1)*tf.square(new_delta)
        delta_rms_var.assign(new_delta_rms)
        grad_rms_var.assign(new_grad_rms)
        var.assign(new_var)



class SGLDOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, sigma=0.01, name="SGOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self.sigma = sigma
#
    def _decayed_sigma(self, var_dtype):
        if isinstance(self.sigma, tf.keras.optimizers.schedules.LearningRateSchedule):
            local_step = tf.cast(self.iterations, var_dtype)
            sigma = tf.cast(self.sigma(local_step), var_dtype)
            return sigma
        else:
            return self.sigma
#
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        sigma = self._decayed_sigma(var_dtype)
        new_var = var - lr_t*grad + sigma*tf.math.sqrt(lr_t)*tf.random.normal(shape=tf.shape(grad))
        var.assign(new_var)
