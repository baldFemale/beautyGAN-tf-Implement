import tensorflow as tf

from layers import *

ngf = 32
ndf = 64


def build_generator(input_A,input_B,name="generator"):
    with tf.variable_scope(name):
        ks = 3
        fs = 7
        input_pad_A = tf.pad(input_A,[[0,0],[3,3],[3,3],[0,0]],"REFLECT")
        input_pad_B = tf.pad(input_B,[[0,0],[3,3],[3,3],[0,0]],"REFLECT")

        A_c1 = generate_conv2d(inputconv=input_pad_A,o_d=ngf,kernal_size=fs,stride=1,padding="VALID",
                               name="A_c1",stddev=0.02)  # 1*256*256*32
        B_c1 = generate_conv2d(inputconv=input_pad_B,o_d=ngf,kernal_size=fs,stride=1,padding="VALID",
                               name="B_c1",stddev=0.02)  # 1*256*256*32
        A_c2 = generate_conv2d(inputconv=A_c1,o_d=ngf*2,kernal_size=ks,stride=2,padding="SAME",
                               name="A_c2",stddev=0.02)  # 1*128*128*64
        B_c2 = generate_conv2d(inputconv=B_c1,o_d=ngf*2,kernal_size=ks,stride=2,padding="SAME",
                               name="B_c2",stddev=0.02)  # 1*128*128*64
        A_c3 = generate_conv2d(inputconv=A_c2,o_d=ngf*4,kernal_size=ks,stride=2,padding="SAME",
                               name="A_c3",stddev=0.02)  # 1*64*64*128
        B_c3 = generate_conv2d(inputconv=B_c2,o_d=ngf*4,kernal_size=ks,stride=2,padding="SAME",
                               name="B_c3",stddev=0.02) #1*64*64*128

        input_res = tf.concat([A_c3,B_c3],axis=-1,name="concat") #1*64*64*256
        o_r1 = generate_resblock(input_res,dim=ngf*8,name="r1")
        o_r2 = generate_resblock(o_r1,dim=ngf*8,name="r2")
        o_r3 = generate_resblock(o_r2,dim=ngf*8,name="r3")
        o_r4 = generate_resblock(o_r3,dim=ngf*8,name="r4")
        o_r5 = generate_resblock(o_r4,dim=ngf*8,name="r5")
        o_r6 = generate_resblock(o_r5,dim=ngf*8,name="r6")
        o_r7 = generate_resblock(o_r6,dim=ngf*8,name="r7")
        o_r8 = generate_resblock(o_r7,dim=ngf*8,name="r8")
        o_r9 = generate_resblock(o_r8,dim=ngf*8,name="r9")

        o_r9_A = tf.slice(o_r9,[0,0,0,0],[1,64,64,128])
        o_r9_B = tf.slice(o_r9,[0,0,0,128],[1,64,64,128])

        A_c4 = generate_deconv2d(inputdeconv=o_r9_A,o_d=ngf*2,kernal_size=ks,stride=2,padding="SAME",
                                 name="A_c4",stddev=0.02)
        B_c4 = generate_deconv2d(inputdeconv=o_r9_B,o_d=ngf*2,kernal_size=ks,stride=2,padding="SAME",
                                 name="B_c4",stddev=0.02)
        A_c5 = generate_deconv2d(inputdeconv=A_c4,o_d=ngf,kernal_size=ks,stride=2,padding="SAME",
                                 name="A_c5",stddev=0.02)
        B_c5 = generate_deconv2d(inputdeconv=B_c4,o_d=ngf,kernal_size=ks,stride=2,padding="SAME",
                                 name="B_c5",stddev=0.02)
        A_c6 = generate_deconv2d(inputdeconv=A_c5,o_d=3,kernal_size=fs,stride=1,padding="SAME",
                                 name="A_c6",stddev=0.02,do_relu=False)
        B_c6 = generate_deconv2d(inputdeconv=B_c5,o_d=3,kernal_size=fs,stride=1,padding="SAME",
                                 name="B_c6",stddev=0.02,do_relu=False)

        out_gen_A = tf.nn.tanh(A_c6,name="A_t")
        out_gen_B = tf.nn.tanh(B_c6,name="B_t")

    return out_gen_A,out_gen_B


def generate_discriminator(inputdis,name="discriminator"):
    """
    :param inputdis: 1*256*256*3
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        f = 4
        # 目前的spectral normlization 有点问题
        oc_1 = generate_conv2d(inputdis,64,f,2,"SAME",name="c1",do_norm=False,relufactor=0.2)  # 1*128*128*64
        oc_2 = generate_conv2d(oc_1,128,f,2,"SAME",name="c2",do_norm=False,do_sp_norm=True,relufactor=0.2)  # 1*64*64*128
        oc_3 = generate_conv2d(oc_2,256,f,2,"SAME",name="c3",do_norm=False,do_sp_norm=True,relufactor=0.2)  # 1*32*32*256
        oc_4 = generate_conv2d(oc_3,512,f,1,"SAME",name="c4",do_norm=False,do_sp_norm=True,relufactor=0.2)  # 1*32*32*512
        oc_5 = generate_conv2d(oc_4,1,f,1,"SAME",name="c5",do_norm=False,do_sp_norm=False,do_relu=False)  # 1*32*32*1
        return oc_5
