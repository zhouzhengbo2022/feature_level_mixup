# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AllConv implementation (https://arxiv.org/abs/1412.6806)."""
import math
import torch
import torch.nn as nn
import numpy as np

class GELU(nn.Module):

  def forward(self, x):
    return torch.sigmoid(1.702 * x) * x


def make_layers(cfg):
  """Create a single layer."""
  layers = []
  in_channels = 3
  for v in cfg:
    if v == 'Md':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.5)]
    elif v == 'A':
      layers += [nn.AvgPool2d(kernel_size=8)]
    elif v == 'NIN':
      conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=1)
      layers += [conv2d, nn.BatchNorm2d(in_channels), GELU()]
    elif v == 'nopad':
      conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)
      layers += [conv2d, nn.BatchNorm2d(in_channels), GELU()]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      layers += [conv2d, nn.BatchNorm2d(v), GELU()]
      in_channels = v
  return nn.Sequential(*layers)

class SELayer(nn.Module): 
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__() 
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential( 
            nn.Linear(channel, channel // reduction, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(channel // reduction, channel, bias=False), 
            nn.Softmax() 
         ) 
 
 
    def forward(self, x): 
        x = x.permute(2,0,1)
        c,h,w = x.size() 
        y = self.avg_pool(x)
        #print(y.size())
        y = y.permute(1,2,0)
        y = self.fc(y)
        #print(y.size())
        x = x.permute(1,2,0)
        return x * y.expand_as(x)


class AllConvNet(nn.Module):
  """AllConvNet main class."""

  def __init__(self, num_classes):
    super(AllConvNet, self).__init__()

    self.num_classes = num_classes
    self.width1, w1 = 96, 96
    self.width2, w2 = 192, 192

    self.features = make_layers(
        [w1, w1, w1, 'Md', w2, w2, w2, 'Md', 'nopad', 'NIN', 'NIN', 'A'])
    self.classifier = nn.Linear(self.width2, num_classes)
    #self.se = SELayer(3,1)
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))  # He initialization
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    self.se = SELayer(3,3)
  def forward(self, x):
    if type(x) == list:
        out_list = []
        ws = np.float32(np.random.dirichlet([1] * 3))
        m = np.float32(np.random.beta(1, 1))
        for num in range(3):
            x[num] = x[num].cuda()
            out = self.features(x[num])
            out = out.view(out.size(0), -1)
            out_list.append(out)
            #print(out.shape)
            if num==0:
                mix = torch.zeros_like(out)
            #mix += ws[num] * out
            #x = self.classifier(x)
        out_all = torch.stack([out_list[0],out_list[1],out_list[2]],2).cuda()
        out_all = self.se(out_all)
        results = torch.split(out_all,1,dim=2)
        #print(out_all.shape)
        for i in results:
            mix += i.squeeze(2)
        x[-1] = x[-1].cuda()
        out_ori = self.features(x[-1])
        out_ori = out_ori.view(out_ori.size(0), -1)
        out_mixed = (1 - m) * out_ori + m * mix
        out_mixed = self.classifier(out_mixed)
    else:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out_mixed = self.classifier(x)
    return out_mixed
