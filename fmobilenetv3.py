import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
# import symbol_utils


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


# def adaptiveAvgPool(inputsz, outputsz):
#     import numpy as np
#     s = np.floor(inputsz/outputsz).astype(np.int32)
#     k = inputsz-(outputsz-1)*s
#     return nn.AvgPool2D((k, k), s)
    

class AdaptiveAvgPool2D(nn.HybridBlock):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2D, self).__init__()
        self.output_size = output_size
    
    def hybrid_forward(self, F, x):
        return F.contrib.AdaptiveAvgPooling2D(x, self.output_size)


class ReLU6(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6)


class HSwish(nn.HybridBlock):
    def __init__(self):
        super(HSwish, self).__init__()
    
    def hybrid_forward(self, F, x):
        return x * F.clip(x+3.0, 0, 6)/ 6.0


class HSigmoid(nn.HybridBlock):
    def __init__(self):
        super(HSigmoid, self).__init__()
    
    def hybrid_forward(self, F, x):
        return F.clip(x+3.0, 0, 6)/6.0


class SEBlock(nn.HybridBlock):
    r"""SEBlock from `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    """

    def __init__(self, channels, cardinality, bottleneck_width, stride,
                 downsample=False, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(group_width//2, kernel_size=1, use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(group_width, kernel_size=3, strides=stride, padding=1,
                                use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels * 4, kernel_size=1, use_bias=False))
        self.body.add(nn.BatchNorm())

        self.se = nn.HybridSequential(prefix='')
        self.se.add(nn.Dense(channels // 4, use_bias=False))
        self.se.add(nn.Activation('relu'))
        self.se.add(nn.Dense(channels * 4, use_bias=False))
        self.se.add(nn.Activation('sigmoid'))

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels * 4, kernel_size=1, strides=stride,
                                          use_bias=False))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.se(w)
        x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x


class SEModule(nn.HybridBlock):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        # self.avg_pool = nn.contrib.AdaptiveAvgPooling2D()
        self.fc = nn.HybridSequential()
        self.fc.add(nn.Conv2D(channel//reduction,kernel_size=1, padding=0, use_bias=False),
                    nn.Activation("relu"),
                    nn.Conv2D(channel,kernel_size=1, padding=0, use_bias=False),
                    HSigmoid())
    
    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.fc(w)
        x = F.broadcast_mul(x, w)
        return x


def conv_bn(channels, filter_size, stride, activation=nn.Activation('relu')):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, 3, stride, 1, use_bias=False),
        nn.BatchNorm(scale=True),
        activation
    )
    return out


def conv_1x1_bn(channels, activation=nn.Activation('relu')):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, 1, 1, 0, use_bias=False),
        nn.BatchNorm(scale=True),
        activation
    )
    return out


class MobileBottleNeck(nn.HybridBlock):
    def __init__(self, channels, kernel, stride, exp, se=False, short_cut = True, act="RE"):
        super(MobileBottleNeck, self).__init__()
        self.out = nn.HybridSequential()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        assert act in ["RE", "HS"]
        padding = (kernel - 1) // 2
        self.short_cut = short_cut

        conv_layer = nn.Conv2D
        norm_layer = nn.BatchNorm
        activation = nn.Activation('relu') if act == "RE" else HSwish()
        if se:
            SELayer = SEModule(exp)
            self.out.add(
                conv_layer(exp, 1, 1, 0, use_bias=False),
                norm_layer(scale=True),
                activation,

                conv_layer(exp, kernel, stride, padding, groups=exp, use_bias=False),
                norm_layer(scale=True),
                ############################
                SELayer,
                ############################
                activation,

                conv_layer(channels, 1, 1, 0, use_bias=False),
                norm_layer(scale=True),
                # SELayer(exp, )
            )
        else:
            self.out.add(
                conv_layer(exp, 1, 1, 0, use_bias=False),
                norm_layer(scale=True),
                activation,

                conv_layer(exp, kernel, stride, padding, groups=exp, use_bias=False),
                norm_layer(scale=True),
                activation,

                conv_layer(channels, 1, 1, 0, use_bias=False),
                norm_layer(scale=True),
            )
    def hybrid_forward(self, F, x):
        return x + self.out(x) if self.short_cut else self.out(x)
    


class MobileNetV3(nn.HybridBlock):
    def __init__(self, classes=1000, width_mult=1.0, mode="large", **kwargs):
        super(MobileNetV3, self).__init__()
        assert mode in ["large", "small"]
        # assert input_size%32 == 0
        # self.w = width_mult
        setting = []
        last_channel = 1280
        input_channel = 16

        if mode=="large":
            setting = [
                # k, exp, c,  se,     nl,  s, short_cut
                [3, 16,  16,  False, 'RE', 1, False],
                [3, 64,  24,  False, 'RE', 2, False],
                [3, 72,  24,  False, 'RE', 1, True],
                [5, 72,  40,  True,  'RE', 2, False],
                [5, 120, 40,  True,  'RE', 1, True],
                [5, 120, 40,  True,  'RE', 1, True],
                [3, 240, 80,  False, 'HS', 2, False],
                [3, 200, 80,  False, 'HS', 1, True],
                [3, 184, 80,  False, 'HS', 1, True],
                [3, 184, 80,  False, 'HS', 1, True],
                [3, 480, 112, True,  'HS', 1, False],
                [3, 672, 112, True,  'HS', 1, True],
                [5, 672, 112, True,  'HS', 1, True],  
                [5, 672, 160, True,  'HS', 2, False],
                [5, 960, 160, True,  'HS', 1, True],
            ]
        else:
            setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2, False],
                [3, 72,  24,  False, 'RE', 2, False],
                [3, 88,  24,  False, 'RE', 1, True],
                [5, 96,  40,  True,  'HS', 2, False],  # stride = 2, paper set it to 1 by error
                [5, 240, 40,  True,  'HS', 1, True],
                [5, 240, 40,  True,  'HS', 1, True],
                [5, 120, 48,  True,  'HS', 1, False],
                [5, 144, 48,  True,  'HS', 1, True],
                [5, 288, 96,  True,  'HS', 2, False],
                [5, 576, 96,  True,  'HS', 1, True],
                [5, 576, 96,  True,  'HS', 1, True],
            ]

        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.layers = [conv_bn(input_channel, 3, 2, activation=HSwish())]

        for kernel_size, exp, channel, se, act, s, short_cut in setting:
            # short_cut = (s == 1)
            output_channel = make_divisible(channel * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.layers.append(MobileBottleNeck(output_channel, kernel_size, s, exp_channel, se, short_cut, act))
        
        if mode == "large":
            last_conv = make_divisible(960 * width_mult)
            self.layers.append(conv_1x1_bn(last_channel, HSwish()))
            self.layers.append(AdaptiveAvgPool2D(output_size=1))
            self.layers.append(HSwish())
            self.layers.append(nn.Conv2D(last_channel, 1, 1, 0))
            self.layers.append(HSwish())
        else:
            last_conv = make_divisible(576 * width_mult)
            self.layers.append(conv_1x1_bn(last_channel, HSwish()))
            self.layers.append(SEModule(last_channel)) 
            self.layers.append(AdaptiveAvgPool2D(output_size=1))
            self.layers.append(HSwish())
            self.layers.append(conv_1x1_bn(last_channel, HSwish()))
        
        self._layers = nn.HybridSequential()
        self._layers.add(*self.layers)
    
    def hybrid_forward(self, F, x):
        return self._layers(x)


def get_symbol(num_classes=256, mode="small", **kwargs):
    net = MobileNetV3(mode=mode)
    data = mx.sym.Variable(name='data')
    data = (data-127.5)
    data = data*0.0078125
    body = net(data)
    import symbol_utils
    body = symbol_utils.get_fc1(body, num_classes, "E")
    return body
