# Improving Weight Initialization of ReLU and Output Layers

Paper Title: Improving Weight Initialization of ReLU and Output Layers

We introduce a data-dependent weight initialization scheme for ReLU and output layers commonly found in modern neural network
architectures. An initial feedforward pass through the network is performed using an initialization set (a subset of the
training data set). Using statistics obtained from this pass, we initialize the weights of the network, so the following
properties are met: 1) weight matrices are orthogonal; 2) ReLU layers produce a predetermined fraction of non-zero activations;
3) the outputs produced by internal layers have a predetermined variance; 4) weights in the last layer are chosen to minimize 
the squared error in the initialization set. We evaluate our method on popular architectures (VGG16, VGG19, and InceptionV3) 
and faster convergence rates are achieved on the ImageNet data set when compared to state-of-the-art initialization 
techniques (LSUV, He, and Glorot).

In this repository, you will find our initialization code (Keras). If you would like to add our initialization technique
to your project, see 'main/simple_example.py' to learn how to do so. It's a fairly simple process.
