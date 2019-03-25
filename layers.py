import tensorflow as tf


def instance_norm(x):
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean,var = tf.nn.moments(x,[1,2],keep_dims=True)
        scale = tf.get_variable(name="scale",shape=[x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02,mean=1.0))
        offset = tf.get_variable(name="offset",shape=[x.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean,tf.sqrt(var+epsilon))+offset
        return out


def lrelu(x,leak=0.2,name="lrelu",alt_relu_impl=True):
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5*(1+leak)
            f2 = 0.5*(1-leak)
            return f1*x+f2*abs(x)
        else:
            return tf.maximum(x,leak*x)


def spectral_norm(x, iteration=1):
    """
    following taki0112's implement
    :param x:
    :param iteration:
    :return:
    """
    with tf.variable_scope("spectral_norm"):
        x_shape = x.shape.as_list()
        w = tf.reshape(x, [-1, x_shape[-1]])
        u = tf.get_variable("u", [1, x_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
        u_hat = u
        v_hat = None

        for i in range(iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_, dim=None)
            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_, dim=None)
        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, [-1]+x_shape[1:])
        return w_norm


def generate_conv2d(inputconv, o_d=64, kernal_size=7, stride=1,
                    padding="VALID", name="conv2d",stddev=0.02,
                    do_relu=True, do_norm=True, do_sp_norm=False,relufactor=0.2
                    ):
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(
            inputconv, o_d, kernal_size, stride, padding,
            activation_fn=None,
            weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer = tf.constant_initializer(0.0)
        )

        if do_norm:
            conv = instance_norm(conv)

        if do_sp_norm:
            conv = spectral_norm(conv)

        if do_relu:
            if relufactor!=0:
                conv = lrelu(conv,relufactor,"lrelu")
            else:
                conv = tf.nn.relu(conv,name="relu")

        return conv


def generate_deconv2d(inputdeconv,o_d=64,kernal_size=7,stride=1,padding="VALID",name="deconv2d",
                      stddev=0.02,do_relu=True,do_norm=True,do_sp_norm=False,relufactor=0.2):
    with tf.variable_scope(name):
        deconv = tf.contrib.layers.conv2d_transpose(inputdeconv,o_d,kernal_size,stride,
                                                    padding,activation_fn=None,
                                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                    biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            deconv = instance_norm(deconv)

        if do_sp_norm:
            deconv = spectral_norm(deconv)

        if do_relu:
            if relufactor!=0:
                deconv = lrelu(deconv,relufactor,name="lrelu")
            else:
                deconv = tf.nn.relu(deconv,name="relu")
        return deconv


def generate_resblock(input_res,dim,name="resnet"):
    with tf.variable_scope(name):
        out_res = tf.pad(input_res,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
        out_res = generate_conv2d(inputconv=out_res,o_d=dim,kernal_size=3,stride=1,padding="VALID",name="c1")
        out_res = tf.pad(out_res,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
        out_res = generate_conv2d(out_res,dim,3,1,"VALID","c2",do_relu=False)
        return tf.nn.relu(input_res+out_res)
