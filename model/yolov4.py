# cls=20, base yolo5

import torch
import torch.nn as nn
import common as C

from torchsummary import summary

class YOLOV4(nn.Module):
    def __init__(self):
        super(YOLOV4, self).__init__()

        self.input = nn.Sequential(
            C.Conv(3, 32, 3, 1, mish_act=True),
        )

        self.group0 = nn.Sequential(
            C.Conv(32, 64, 3, 2, mish_act=True),
            C.BottleneckCSP(64, 64, mish_csp=True),
        )

        self.group1 = nn.Sequential(
            C.Conv(64, 128, 3, 2, mish_act=True),
            C.BottleneckCSP(128, 128, mish_csp=True),
            C.BottleneckCSP(128, 128, mish_csp=True),
        )

        self.group2 = nn.Sequential(
            C.Conv(128, 256, 3, 2, mish_act=True),
            C.BottleneckCSP(256, 256, mish_csp=True),
            C.BottleneckCSP(256, 256, mish_csp=True),
            C.BottleneckCSP(256, 256, mish_csp=True),
            C.BottleneckCSP(256, 256, mish_csp=True),
            C.BottleneckCSP(256, 256, mish_csp=True),
            C.BottleneckCSP(256, 256, mish_csp=True),
            C.BottleneckCSP(256, 256, mish_csp=True),
            C.BottleneckCSP(256, 256, mish_csp=True),
        )  # route -21

        self.group3 = nn.Sequential(
            C.Conv(256, 512, 3, 2, mish_act=True),
            C.BottleneckCSP(512, 512, mish_csp=True),
            C.BottleneckCSP(512, 512, mish_csp=True),
            C.BottleneckCSP(512, 512, mish_csp=True),
            C.BottleneckCSP(512, 512, mish_csp=True),
            C.BottleneckCSP(512, 512, mish_csp=True),
            C.BottleneckCSP(512, 512, mish_csp=True),
            C.BottleneckCSP(512, 512, mish_csp=True),
            C.BottleneckCSP(512, 512, mish_csp=True),
        )  #route　-10
        
        self.group4 = nn.Sequential(
            C.Conv(512, 1024, 3, 2, mish_act=True),
            C.BottleneckCSP(1024, 1024, mish_csp=True),
            C.BottleneckCSP(1024, 1024, mish_csp=True),
            C.BottleneckCSP(1024, 1024, mish_csp=True),
            C.BottleneckCSP(1024, 1024, mish_csp=True),
        )

        self.spp = nn.Sequential(
            C.Conv(1024, 512, 1, 1),
            C.Conv(512, 1024, 3, 1),
            C.SPP(1024, 512, k=[5, 9, 13]),
            C.Conv(512, 1024, 3, 1),
            C.Conv(1024, 512, 1, 1),
        )  # route 15

        ##################################################
        ############# backbone -> neck -> head############
        ##################################################

        self.neck0 = C.Conv(512, 256, 1, 1)
        self.neck0UP = nn.Upsample(scale_factor=2, mode='nearest')
        self.neck0route = C.Conv(512, 256, 1, 1)
        '''
        Up
        route
        concat
        out 38*38*512
        '''
        self.neck1 = nn.Sequential(
            C.Conv(512, 256, 1, 1),
            C.Conv(256, 512, 3, 1),
            C.Conv(512, 256, 1, 1),
            C.Conv(256, 512, 3, 1),
            C.Conv(512, 256, 1, 1),
        )

        self.neck2 = C.Conv(256, 128, 1, 1)
        self.neck2UP = nn.Upsample(scale_factor=2, mode='nearest')
        self.neck2route = C.Conv(256, 128, 1, 1)
        '''
        Up
        route
        concat
        out 76*76*256
        '''

        self.neck3 = nn.Sequential(
            C.Conv(256, 128, 1, 1),
            C.Conv(128, 256, 3, 1),
            C.Conv(256, 128, 1, 1),
            C.Conv(128, 256, 3, 1),
            C.Conv(256, 128, 1, 1),
        )

        self.head0 = nn.Sequential(
            C.Conv(128, 256, 3, 1),
            nn.Conv2d(256, 75, 1, 1),
        )

        self.neck3route = C.Conv(128, 256, 3, 2)
        '''
        route
        concat
        out:38*38*512
        '''

        self.neck4 = nn.Sequential(
            C.Conv(512, 256, 1, 1),
            C.Conv(256, 512, 3, 1),
            C.Conv(512, 256, 1, 1),
            C.Conv(256, 512, 3, 1),
            C.Conv(512, 256, 1, 1),
        )  #route -3

        self.head1 = nn.Sequential(
            C.Conv(256, 512, 3, 1),
            nn.Conv2d(512, 75, 1, 1),
        )

        self.neck4route = C.Conv(256, 512, 3, 2)
        '''
        route
        concat
        out:19*19*1024
        '''

        self.neck5 = nn.Sequential(
            C.Conv(1024, 512, 1, 1),
            C.Conv(512, 1024, 3, 1),
            C.Conv(1024, 512, 1, 1),
            C.Conv(512, 1024, 3, 1),
            C.Conv(1024, 512, 1, 1),
        )  

        self.head2 = nn.Sequential(
            C.Conv(512, 1024, 3, 1),
            nn.Conv2d(1024, 75, 1, 1),
        )

    def forward(self, x):
        x0 = self.input(x)
        x1 = self.group0(x0)
        x2 = self.group1(x1)
        x3 = self.group2(x2)
        x4 = self.group3(x3)
        x5 = self.group4(x4)
        x6 = self.spp(x5)
        
        x7 = self.neck0(x6)
        x8 = self.neck0UP(x7)
        x9 = self.neck0route(x4) 
        x10 = torch.cat((x9, x8), 1)
        
        x11 = self.neck1(x10)
        
        x12 = self.neck2(x11)
        x13 = self.neck2UP(x12)
        x14 = self.neck2route(x3)
        x15 = torch.cat((x14, x13), 1)
        
        x16 = self.neck3(x15)
        head0 = self.head0(x16)

        x17 = self.neck3route(x16)
        x18 = torch.cat((x17, x11), 1)

        x19 = self.neck4(x18)
        head1 = self.head1(x19)

        x20 = self.neck4route(x19)
        x21 = torch.cat((x20, x6), 1)

        x22 = self.neck5(x21)
        head2 = self.head2(x22)

        head0 = head0.view(head0.size(0), -1)
        head1 = head1.view(head1.size(0), -1)
        head2 = head2.view(head2.size(0), -1)
        return head0, head1, head2

if __name__ == "__main__":
    img = torch.rand((1, 3, 416, 416), dtype = torch.float32)
    net = YOLOV4()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    print(model)
    summary(model, (3, 416, 416))
