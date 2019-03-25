# beautyGAN-tf-Implement
* A tensorflow Implement for [BeautyGAN:Instance-level facial makeup transfer with deep generative adversarial networks](http://liusi-group.com/projects/BeautyGAN).
* Based on [cycleGAN tf Implement](https://github.com/hardikbansal/CycleGAN)
* [Baidu Drive for vgg16.npy](https://pan.baidu.com/s/1D4Zoaunwo2rZTNW7HhZjPA): 83dt
* face parsing tools: [dlib'68 landmarks model](http://dlib.net/files/)
* remaining questions:
  * histogram match on shadow instead of eyes(Now we just ignore the histogram loss on eyes)
  * channel(generator channel number decreased & 70X70 patch discriminator bug fixed)
