import torch
import numpy as np
import pdb
import logging
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from distutils import util
import torch.nn.functional as F


class LSTM_SE_PL_Dense_MTL(nn.Module):
    def __init__(self, fea_dim, context_len, hidden_dim, hidden_layers, output_dim, bidirectional ):
        super(LSTM_SE_PL_Dense_MTL, self).__init__()
        self.fea_dim = fea_dim
        self.input_size = fea_dim * context_len
        self.hidden_size = hidden_dim
        self.num_layers = hidden_layers
        self.bidirectional= bool(util.strtobool(bidirectional))
        if self.bidirectional:
            num_direction = 2
        else:
            num_direction =1

        self.stack_rnn_stage1 = nn.LSTM(self.input_size, hidden_size=hidden_dim,num_layers=1, bidirectional=self.bidirectional, dropout=0)
        self.fconn_lps_stage1 = nn.Linear( hidden_dim * num_direction, output_dim)
        self.fconn_mask_stage1 = nn.Linear( hidden_dim * num_direction, output_dim) 

        # dense-layer makes the fea_dim be multiplied by 2
        self.stack_rnn_stage2 = nn.LSTM(fea_dim * 2, hidden_size=hidden_dim,num_layers=1, bidirectional=self.bidirectional, dropout=0)
        self.fconn_lps_stage2 = nn.Linear( hidden_dim * num_direction, output_dim)
        self.fconn_mask_stage2 = nn.Linear( hidden_dim * num_direction, output_dim) 

        # dense-layer makes the fea_dim be multiplied by 3
        self.stack_rnn_stage3 = nn.LSTM(fea_dim * 3, hidden_size=hidden_dim,num_layers=1, bidirectional=self.bidirectional, dropout=0)
        self.fconn_lps_stage3 = nn.Linear( hidden_dim * num_direction, output_dim)
        self.fconn_mask_stage3 = nn.Linear( hidden_dim * num_direction, output_dim)


    def forward(self, input_features):
        lps_outputs = []
        irm_outputs = []

        output, (_,_) = self.stack_rnn_stage1(input_features )
        estimated_lps_stage1 =  self.fconn_lps_stage1(output) 
        estimated_irm_stage1 = torch.sigmoid ( self.fconn_mask_stage1(output) )

        lps_outputs.append( estimated_lps_stage1 )
        irm_outputs.append( estimated_irm_stage1 )

        # concat the frame of input_features and the estimated feature together, as the input for the next layer to compensate for information loss
        input_frame = input_features[:, :, self.fea_dim * 3:self.fea_dim * 4 ]
        cur_inputs = torch.cat((input_frame, lps_outputs[0] ), -1)

        output, (_,_) = self.stack_rnn_stage2(cur_inputs )
        estimated_lps_stage2 = self.fconn_lps_stage2(output)
        estimated_irm_stage2 = torch.sigmoid (self.fconn_mask_stage2(output) )
        lps_outputs.append( estimated_lps_stage2 )
        irm_outputs.append( estimated_irm_stage2 )

        cur_inputs = torch.cat((cur_inputs, lps_outputs[1] ), -1)
        output, (_,_) = self.stack_rnn_stage3(cur_inputs )
        estimated_lps_stage3 = self.fconn_lps_stage3(output)
        estimated_irm_stage3 = torch.sigmoid ( self.fconn_mask_stage3(output) )
        lps_outputs.append( estimated_lps_stage3 )
        irm_outputs.append( estimated_irm_stage3 )


        return lps_outputs, irm_outputs







