# beautyGAN-tf-Implement
* A tensorflow Implement for [BeautyGAN:Instance-level facial makeup transfer with deep generative adversarial networks](http://liusi-group.com/projects/BeautyGAN).
* Based on [cycleGAN tf Implement](https://github.com/hardikbansal/CycleGAN)
* [Baidu Drive for vgg16.npy](https://pan.baidu.com/s/1D4Zoaunwo2rZTNW7HhZjPA): 83dt
* face parsing tools: [dlib'68 landmarks model](http://dlib.net/files/)

# question
* face parsing is not precise -> remove face mask/ add total variation loss

# attention
* in order to speed up the training process, we fix the (content,style) pair. remove this constraint should improve the model's performance on validation data and test data.
* we simply apply the face parser in opencv, the original paper use different face parsing tool.

# some results
![image](https://github.com/baldFemale/beautyGAN-tf-Implement/raw/master/results/results.png)
