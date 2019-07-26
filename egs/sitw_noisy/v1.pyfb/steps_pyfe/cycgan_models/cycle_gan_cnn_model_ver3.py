import torch
import torch.nn as nn
import torch.nn.functional as F

class CycleGANCnnModelVer3(object):
    def __init__(self, train_args):
        '''
            SRC Domain (NB) input size is SEQ_LEN x 40
            TGT Domain (WB) input size is SEQ_LEN x 40
            Accepted SEQ_LENs: 11, 127
            SRC and TGT domain have same dimensions
            hence IDT loss can be applied with this model
            the generator output is x + self.model(x) (system C in ICASSP model)
            Disc is fully convolutional that is similar to the one in cycle GAN network
            (c4s2-64-LR, c4s2-128-LR, c4s2-256-LR, c4s1-512-LR, c4s1-1)
            This version is similar to Ver3 except for the difference in discriminators
            Ver3 disc is 2convs followed by 2 FL layers
            Ver4 is fully convolutional
        '''
        networks = self.build_networks(**(vars(train_args)))
        self.src2tgt_gen, self.tgt2src_gen, self.src_disc, self.tgt_disc = networks

    def build_networks(self, nc, gen_nfilter, gen_num_res_blocks,
                            reg_type, gen_kernel_size_first_layer,
                            gen_padding_type, dropout_per, use_norm_layer,
                            norm_layer_type, hid_act_type, disc_hid_act_type,
                            use_lsgan, src_name, tgt_name,
                            disc_nfilter, disc_kernel_size, **kwargs):

        lsgan = use_lsgan

        # Build src2tgt_gen
        src2tgt_gen = ConvGenNet(nc, nc, gen_nfilter, gen_num_res_blocks,
                                        reg_type, gen_kernel_size_first_layer,
                                        gen_padding_type, dropout_per, use_norm_layer,
                                        norm_layer_type, hid_act_type)

        # Build tgt2src_gen
        tgt2src_gen = ConvGenNet(nc, nc, gen_nfilter, gen_num_res_blocks,
                                        reg_type, gen_kernel_size_first_layer,
                                        gen_padding_type, dropout_per, use_norm_layer,
                                        norm_layer_type, hid_act_type)

        # src disc
        src_disc = ConvDiscNet(nc, disc_hid_act_type, lsgan=lsgan,
                    disc_nfilter=disc_nfilter, disc_kernel_size=disc_kernel_size)

        # tgt disc
        tgt_disc = ConvDiscNet(nc, disc_hid_act_type, lsgan=lsgan,
                    disc_nfilter=disc_nfilter, disc_kernel_size=disc_kernel_size)


        return src2tgt_gen, tgt2src_gen, src_disc, tgt_disc


class ConvGenNet(nn.Module):
    def __init__(self, input_nc, output_nc, nfilter, num_res_blocks,
                        reg_type, kernel_size_first_layer, padding_type,
                        dropout_per, use_norm_layer,
                        norm_layer_type, hid_act_type, **kwargs):
        super(ConvGenNet,self).__init__()
        self.model = self.build_network(input_nc, output_nc, nfilter, num_res_blocks,
                                        reg_type, kernel_size_first_layer, padding_type,
                                        dropout_per, use_norm_layer, norm_layer_type,
                                        hid_act_type, **kwargs)
    '''
    Similar to the one proposed in original Cycle GAN model
    INP: SRC (or TGT) domain features of input (N, 1, SEQ_LEN, 40)
    OUT: TGT (or SRC) domain features of input (N, 1, SEQ_LEN, 40)
    '''
    def build_network(self, input_nc, output_nc, nfilter, num_res_blocks, reg_type,
                    kernel_size_first_layer, padding_type,
                    dropout_per, use_norm_layer, norm_layer_type, hid_act_type, **kwargs):

        pad_dim = (kernel_size_first_layer-1)//2
        kernel_size = 3    # Hard coded

        # First conv layer
        use_dropout = (reg_type == 'dropout') or (reg_type == 'dropout2d')
        if use_dropout:
            if reg_type == 'dropout':
                self.dropout_layer = nn.Dropout
            elif reg_type == 'dropout2d':
                self.dropout_layer = nn.Dropout2d
        else:
            self.dropout_layer = None


        # Get batch norm layer
        if norm_layer_type == 'batchnorm2d':
            norm_layer = nn.BatchNorm2d
        elif norm_layer_type == 'instancenorm2d':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('norm layer {n} not implemented yet'.format(
                                        n=norm_layer_type))

        # Get padding layer
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        elif padding_type == 'replicate':
            padding_layer = nn.ReplicationPad2d

        # Get activation layer
        if hid_act_type == 'relu':
            hid_act_layer = nn.ReLU()
        elif hid_act_type == 'prelu':
            hid_act_layer = nn.PReLU(init=0.2)
        elif hid_act_type == 'leakyrelu':
            hid_act_layer = nn.LeakyReLU(0.2)
        else:
            raise NotImplementedError('hid act type {n} not implemented yet'.format(
                                        n=hid_act_type))

        layers = []
        layers.append(padding_layer(pad_dim))
        layers.append(nn.Conv2d(input_nc, nfilter,
                        (kernel_size_first_layer, kernel_size_first_layer),
                        (1, 1), padding = (0,2)))
        layers.append(hid_act_layer)
        if use_dropout:
            layers.append(self.dropout_layer(dropout_per))

        # Second conv layer
        layers.append(nn.Conv2d(nfilter, nfilter*2, (kernel_size, kernel_size),
                                    (2, 2), padding = 1))
        if use_norm_layer:
            layers.append(norm_layer(nfilter*2))
        layers.append(hid_act_layer)
        if use_dropout:
            layers.append(self.dropout_layer(dropout_per))

        # Third conv layer
        layers.append(nn.Conv2d(nfilter*2, nfilter*4, (kernel_size, kernel_size),
                                    (2, 2), padding = (1,0)))
        if use_norm_layer:
            layers.append(norm_layer(nfilter*4))
        layers.append(hid_act_layer)
        if use_dropout:
            layers.append(self.dropout_layer(dropout_per))

        # Resblocks
        res_padding_type = 'reflect'
        for res_block_id in range(num_res_blocks):
            layers.append(ResnetBlock(nfilter * 4, padding_type=res_padding_type,
                                use_norm_layer=use_norm_layer, norm_layer=norm_layer,
                                hid_act_layer=hid_act_layer,
                                use_dropout=use_dropout,
                                dropout_layer=self.dropout_layer, dropout_per=dropout_per,
                                use_bias=True))
            #layers.append(norm_layer(nfilter*4))
            layers.append(hid_act_layer)
            if use_dropout:
                layers.append(self.dropout_layer(dropout_per))


        # First deconv layer
        layers.append(nn.ConvTranspose2d(nfilter*4, nfilter*2,
                                        (kernel_size, kernel_size), (2, 2),
                                        padding=1, output_padding=(1, 1)))
        if use_norm_layer:
            layers.append(norm_layer(nfilter*2))
        layers.append(hid_act_layer)
        if use_dropout:
            layers.append(self.dropout_layer(dropout_per))


        # Second deconv layer
        layers.append(nn.ConvTranspose2d(nfilter*2, nfilter,
                                        (kernel_size, kernel_size), (2, 2),
                                        padding=(1,1), output_padding=(0, 1)))

        if use_norm_layer:
            layers.append(norm_layer(nfilter))
        layers.append(hid_act_layer)
        if use_dropout:
            layers.append(self.dropout_layer(dropout_per))

        # Last conv layer
        layers.append(padding_layer(pad_dim))
        layers.append(nn.Conv2d(nfilter, output_nc,
                            (kernel_size_first_layer, kernel_size_first_layer),
                            (1, 1), padding=0))

        model = nn.Sequential(*layers)

        return model


    def forward(self, x):
        o = x + self.model(x)

        return o


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, use_norm_layer,
                        norm_layer, hid_act_layer, use_dropout,
                        dropout_layer, dropout_per, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_norm_layer,
                                    norm_layer, hid_act_layer,
                                    use_dropout, dropout_layer, dropout_per, use_bias)

    def build_conv_block(self, dim, padding_type, use_norm_layer, norm_layer,
                                hid_act_layer, use_dropout,
                                dropout_layer, dropout_per, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       hid_act_layer]

        if use_dropout:
            conv_block += [dropout_layer(dropout_per)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        if use_norm_layer:
            conv_block += [norm_layer(dim)]

        return nn.Sequential(*conv_block)


    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ConvDiscNet(nn.Module):
    def __init__(self, input_nc, hid_act_type,
                        norm_layer_type='instancenorm2d', lsgan=False,
                        disc_nfilter=64, disc_kernel_size=4):
        '''
        Cycle GAN discriminator (Full convolutional)
        c4s2-64-LR, c4s2-128-LR, c4s2-256-LR, c4s1-512-LR, c4s1-1.
        Takes as input NB features of dimension (N, 1, 127, 40)
        '''
        super(ConvDiscNet, self).__init__()
        use_norm = False

        # Get activation layer
        if hid_act_type == 'relu':
            hid_act_layer = nn.ReLU()
        elif hid_act_type == 'prelu':
            hid_act_layer = nn.PReLU(init=0.2)
        elif hid_act_type == 'leakyrelu':
            hid_act_layer = nn.LeakyReLU(0.2)
        else:
            raise NotImplementedError('hid act type {n} not implemented yet'.format(
                                        n=hid_act_type))

        # Get batch norm layer
        if norm_layer_type == 'batchnorm2d':
            norm_layer = nn.BatchNorm2d
        elif norm_layer_type == 'instancenorm2d':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('norm layer {n} not implemented yet'.format(
                                        n=norm_layer_type))


        # First conv layer
        layers = []
        layers.append(nn.Conv2d(input_nc, disc_nfilter, disc_kernel_size, 2, padding=1))
        if use_norm:
            layers.append(norm_layer(disc_nfilter))
        layers.append(hid_act_layer)

        # Second conv layer
        layers.append(nn.Conv2d(disc_nfilter, disc_nfilter*2,
                                disc_kernel_size, 2, padding=1))
        if use_norm:
            layers.append(norm_layer(disc_nfilter*2))
        layers.append(hid_act_layer)

        # Third conv layer
        layers.append(nn.Conv2d(disc_nfilter*2, disc_nfilter*4,
                                disc_kernel_size, 2, padding=1))
        if use_norm:
            layers.append(norm_layer(disc_nfilter*4))
        layers.append(hid_act_layer)

        # Fourth conv layer
        layers.append(nn.Conv2d(disc_nfilter*4, disc_nfilter*8,
                                disc_kernel_size, 1, padding=1))
        if use_norm:
            layers.append(norm_layer(disc_nfilter*8))
        layers.append(hid_act_layer)

        # Final conv layer
        layers.append(nn.Conv2d(disc_nfilter*8, 1, disc_kernel_size, 1, padding=1))

        # Conv net
        self.conv_net = nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv_net(x)
        return x
