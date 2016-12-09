# GAN Resize Convolution

Comparison between a regular GAN and the *resize-convolution* propsed in the article [http://distill.pub/2016/deconv-checkerboard/](http://distill.pub/2016/deconv-checkerboard/) as an alternative to the standard deconvolution (transposed convolution) in the generator to get rid of the checkboard artifacts.

## Upsampling

The upsampling was made using the [2d unpooling](http://docs.chainer.org/en/stable/reference/functions.html#chainer.functions.unpooling_2d) function in Chainer which is very similar to a [nearest-neighbor interpolation](https://en.wikipedia.org/wiki/Image_scaling).

## Dataset

Training dataset of Cifar10.

## Results

We see that the artifacts are less noticeable in the resize-convolutions and that the images look more natural, but that the difference becomes less obvious as the training goes on.

Left images are images generated using a regular GAN and right images are generated using resize-convolutions. 

#### 1 Iteration (Left: Regular GAN, Right: Resize-convolution)

<img src="./samples/deconv/0.png" width="384px;"/>
<img src="./samples/resize-conv/0.png" width="384px;"/>

#### 1 Epochs

<img src="./samples/deconv/1.png" width="384px;"/>
<img src="./samples/resize-conv/1.png" width="384px;"/>

#### 2 Epochs

<img src="./samples/deconv/2.png" width="384px;"/>
<img src="./samples/resize-conv/2.png" width="384px;"/>

#### 3 Epochs

<img src="./samples/deconv/3.png" width="384px;"/>
<img src="./samples/resize-conv/3.png" width="384px;"/>

#### 4 Epochs

<img src="./samples/deconv/4.png" width="384px;"/>
<img src="./samples/resize-conv/4.png" width="384px;"/>

#### 5 Epochs

<img src="./samples/deconv/5.png" width="384px;"/>
<img src="./samples/resize-conv/5.png" width="384px;"/>

#### 6 Epochs

<img src="./samples/deconv/6.png" width="384px;"/>
<img src="./samples/resize-conv/6.png" width="384px;"/>

#### 7 Epochs

<img src="./samples/deconv/7.png" width="384px;"/>
<img src="./samples/resize-conv/7.png" width="384px;"/>

#### 8 Epochs

<img src="./samples/deconv/8.png" width="384px;"/>
<img src="./samples/resize-conv/8.png" width="384px;"/>

#### 9 Epochs

<img src="./samples/deconv/9.png" width="384px;"/>
<img src="./samples/resize-conv/9.png" width="384px;"/>

#### 10 Epochs

<img src="./samples/deconv/10.png" width="384px;"/>
<img src="./samples/resize-conv/10.png" width="384px;"/>
