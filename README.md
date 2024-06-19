## This code is the source code implementation for the paper "APB-FedAR: Personalized Federated Learning Based on Adaptive Weights and Privacy Budget Allocation".



## Abstract

![](/pic/arc.png)

Federated Learning (FL) is a distributed learning that does not require the concentration of raw data on servers, but its learning performance is greatly challenged due to the non-independent and identically distributed (Non-IID) nature of data. Personalized Federated Learning (PFL) is used to solve this problem, but existing PFL methods often sacrifice the performance of users with small amounts of data, resulting in unfair global models. On the other hand, although Differential Privacy Federated Learning can address user privacy issues, the addition of noise can hurt the accuracy and fairness of the model. To address these two issues, we propose a PFL framework based on adaptive weights (FedAR). Using an exponential forgetting mechanism to evaluate historical losses, and then calculating aggregate weights based on historical losses to address the issue of global model unfairness. Secondly, we proposed a PFL framework based on adaptive privacy budget allocation (APB-FedAR). By carefully allocating user privacy budgets, we aim to reduce the negative impact of noise on the model. Finally, we conducted extensive experiments on three real datasets to demonstrate the effectiveness of FedAR and APB-FedAR.



## Experimental Environment

**Operating environment：**

GPU：NVIDIA A100 SXM4 40G GPU 

CPU：Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz

**Installation：**

To run the code, you need to install the following packages：

```
absl-py	1.4.0
backpack	0.1
blas	1.0
bottleneck	1.3.5
brotli	1.0.9
brotli-bin	1.0.9
brotlipy	0.7.0
bzip2	1.0.8
ca-certificates	2023.01.10
cachetools	5.3.0
calmsize	0.1.3
certifi	2022.12.7
cffi	1.15.1
charset-normalizer	2.0.4
colorama	0.4.6
contourpy	1.0.5
cryptography	38.0.1
cudatoolkit	11.3.1
cycler	0.11.0
dill	0.3.6
easydict	1.10
fftw	3.3.9
flit-core	3.6.0
fonttools	4.25.0
freetype	2.12.1
functorch	1.13.0
future	0.18.2
glib	2.69.1
google-auth	2.16.2
google-auth-oauthlib	0.4.6
grpcio	1.51.3
gst-plugins-base	1.18.5
gstreamer	1.18.5
h5py	3.7.0
hdf5	1.10.6
icc_rt	2022.1.0
icu	58.2
idna	3.4
intel-openmp	2021.4.0
joblib	1.1.1
jpeg	9e
keras	2.12.0
kiwisolver	1.4.4
lerc	3.0
libbrotlicommon	1.0.9
libbrotlidec	1.0.9
libbrotlienc	1.0.9
libclang	12.0.0
libdeflate	1.8
libffi	3.4.2
libiconv	1.16
libogg	1.3.5
libpng	1.6.37
libtiff	4.4.0
libuv	1.40.0
libvorbis	1.3.7
libwebp	1.2.4
libwebp-base	1.2.4
libxml2	2.9.14
libxslt	1.1.35
lz4-c	1.9.4
markdown	3.4.1
markupsafe	2.1.2
matplotlib	3.7.1
matplotlib-base	3.7.1
mkl	2021.4.0
mkl-service	2.4.0
mkl_fft	1.3.1
mkl_random	1.2.2
munkres	1.1.4
ninja	1.10.2
ninja-base	1.10.2
numexpr	2.8.4
numpy	1.23.4
numpy-base	1.23.4
oauthlib	3.2.2
opacus	1.3.0
openssl	1.1.1t
opt-einsum	3.3.0
packaging	23.0
pandas	1.5.3
pcre	8.45
pillow	9.2.0
pip	22.3.1
ply	3.11
protobuf	4.22.1
pyasn1	0.4.8
pyasn1-modules	0.2.8
pycparser	2.21
pyopenssl	22.0.0
pyparsing	3.0.9
pyqt	5.15.7
pyqt5-sip	12.11.0
pysocks	1.7.1
python	3.10.8
python-dateutil	2.8.2
pytorch	1.12.0
pytorch-mutex	1.0
pytz	2022.7
pyyaml	6.0
qt-main	5.15.2
qt-webengine	5.15.9
qtwebkit	5.212
requests	2.28.1
requests-oauthlib	1.3.1
rsa	4.9
scikit-learn	1.1.3
scipy	1.9.3
setuptools	65.5.0
simplejson	3.19.1
sip	6.6.2
six	1.16.0
sqlite	3.40.0
tensorboard	2.12.0
tensorboard-data-server	0.7.0
tensorboard-plugin-wit	1.8.1
threadpoolctl	2.2.0
tk	8.6.12
toml	0.10.2
torchaudio	0.12.0
torchsummary	1.5.1
torchvision	0.13.0
tornado	6.2
tqdm	4.64.1
typing-extensions	4.4.0
typing_extensions	4.4.0
tzdata	2022g
ujson	5.4.0
urllib3	1.26.13
vc	14.2
vs2015_runtime	14.27.29016
werkzeug	2.2.3
wheel	0.37.1
win_inet_pton	1.1.0
wincertstore	0.2
xz	5.2.8
yaml	0.2.5
zlib	1.2.13
zstd	1.5.2

```

## Datasets

```
CIFAR-10
CIFAR-100
MNIST
```

## Experimental Setup

**Hyperparameters:**

- Training is conducted over 100 rounds with 10, 20, and 30 clients participating. Each client executes 4 epochs per round.
- The local learning rate is set at 0.01, and the batch size is 128.

**Models:**

- ResNet-34 and MobileNet-V1 are used for evaluation.
- ResNet-34 has 34 convolutional layers, including 1 initial convolutional layer, 4 sets of residual blocks, a global average pooling layer, and a fully connected layer.
- MobileNet-V1 consists of sequence convolution and 1×1 convolution, including 13 depth-separable convolution layers, using 3×3 convolution and batch normalization followed by the ReLU activation function.

**Privacy-Preserving Methods:**

- Differential Privacy (DP): Noise is added to the data to protect user privacy while maintaining model accuracy.
- Adaptive Privacy Budget Allocation: Privacy budgets are allocated dynamically based on training progress to reduce the impact of noise on model updates and fairness.

**Evaluation Metrics:**

- **Model Performance:** The accuracy of local and global models is used as the primary metric.
- **Fairness:** Variance of the accuracy distribution of the global model on user local data is used to measure fairness.
- **Privacy Protection:** Privacy is assessed by ensuring the method satisfies differential privacy requirements.

## Experimental Results

The experimental results demonstrate that FedAR significantly improves global model accuracy and fairness across CIFAR-10, CIFAR-100, and MNIST datasets compared to baseline methods. Using ResNet-34 and MobileNet-V1, FedAR achieves superior global model performance while maintaining high fairness, as evidenced by the lowest variance in global model accuracy among all methods. Additionally, the APB-FedAR framework effectively reduces the impact of noise on model updates through adaptive privacy budget allocation, ensuring high accuracy and fairness under privacy constraints. Overall, APB-FedAR provides a robust solution for achieving balanced performance and privacy in federated learning environments.

![](/pic/1.png)

![](/pic/2.png)

![](/pic/3.png)



## Update log

```
- {24.06.16} Uploaded overall framework code and readme file
```

