from chainer import Chain
from chainer import functions as F
from chainer import links as L


class Generator(Chain):
    def __init__(self):
        super().__init__(
            fc=L.Linear(None, 256*2*2),
            dc1=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),
            dc2=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            dc3=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            dc4=L.Deconvolution2D(32, 3, 4, stride=2, pad=1),
            bn0=L.BatchNormalization(256*2*2),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(32)
        )

    def __call__(self, z, test=False):
        h = F.relu(self.bn0(self.fc(z), test=test))
        h = F.reshape(h, (z.shape[0], 256, 2, 2))

        h = F.relu(self.bn1(self.dc1(h), test=test))

        h = F.relu(self.bn2(self.dc2(h), test=test))

        h = F.relu(self.bn3(self.dc3(h), test=test))

        h = F.sigmoid(self.dc4(h))

        return h


class GeneratorResizeConvolution(Chain):
    def __init__(self):
        super().__init__(
            fc=L.Linear(None, 256*2*2),
            c1=L.Convolution2D(256, 128, 3, stride=1, pad=1),
            c2=L.Convolution2D(128, 64, 3, stride=1, pad=1),
            c3=L.Convolution2D(64, 32, 3, stride=1, pad=1),
            c4=L.Convolution2D(32, 3, 3, stride=1, pad=1),
            bn0=L.BatchNormalization(256*2*2),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(32)
        )

    def __call__(self, z, test=False):
        h = F.relu(self.bn0(self.fc(z), test=test))
        h = F.reshape(h, (z.shape[0], 256, 2, 2))

        # Upsample the image and then apply a dimension preserving convolution
        # with a kernel of size (3, 3), stride and padding of size (1, 1).
        h = F.unpooling_2d(h, 2, 2, cover_all=False)
        h = F.relu(self.bn1(self.c1(h), test=test))

        h = F.unpooling_2d(h, 2, 2, cover_all=False)
        h = F.relu(self.bn2(self.c2(h), test=test))

        h = F.unpooling_2d(h, 2, 2, cover_all=False)
        h = F.relu(self.bn3(self.c3(h), test=test))

        h = F.unpooling_2d(h, 2, 2, cover_all=False)
        h = F.sigmoid(self.c4(h))

        return h


class Discriminator(Chain):
    def __init__(self):
        super().__init__(
            c0=L.Convolution2D(None, 3, 3, stride=1, pad=1),
            c1=L.Convolution2D(3, 32, 4, stride=2, pad=1),
            c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            c3=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            c4=L.Convolution2D(128, 256, 4, stride=2, pad=1),
            bn1=L.BatchNormalization(32),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(128),
            bn4=L.BatchNormalization(256),
            fc=L.Linear(None, 2)
        )

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h), test=test))
        h = F.leaky_relu(self.bn2(self.c2(h), test=test))
        h = F.leaky_relu(self.bn3(self.c3(h), test=test))
        h = F.leaky_relu(self.bn4(self.c4(h), test=test))
        h = self.fc(h)
        return h
