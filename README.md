# Neural Tangent Generalization Attacks (NTGA)
[**ICML 2021 Video**]()
| [**Paper**]()
| [**Install Guide**](#installation)
| [**Quickstart**](#usage)
| [**Results**](#results)
| [**Unlearnable Datasets**](#unlearnable-datasets)

![Last Commit](https://img.shields.io/github/last-commit/lionelmessi6410/ntga?color=red)
[![License](https://img.shields.io/github/license/lionelmessi6410/ntga)](https://github.com/lionelmessi6410/ntga/blob/main/LICENSE)

## Overview
This is the repo for [Neural Tangent Generalization Attacks](), Chia-Hung Yuan and Shan-Hung Wu, In Proceedings of ICML 2021.

We study generalization attacks, a new direction for poisoning attacks, where an attacker aims to modify training data in order to spoil the training process such that a trained network lacks generalizability. We devise Neural Tangent Generalization Attacks (NTGAs), a first efficient work enabling clean-label, black-box generalization attacks against Deep Neural Networks.

NTGA declines the generalization ability sharply, i.e. 99% -> 25%, 92% -> 33%, 99% -> 72% on MNIST, CIFAR10 and 2- class ImageNet, respectively. Please see [Results](#results) or the [main paper]() for a more complete results. We also release the **unlearnable** MNIST, CIFAR-10, and 2-class ImageNet generated by NTGA, which can be found and downloaded in [Unlearnable Datasets](#unlearnable-datasets).

## Installation
Our code uses the [Neural Tangents](https://github.com/google/neural-tangents) library, which is built on top of [JAX](https://github.com/google/jax), and [TensorFlow 2.0](https://www.tensorflow.org/). To use JAX with GPU, please follow [JAX's](https://github.com/google/jax/#installation) GPU installation instructions. Otherwise, install JAX on CPU by running

```bash
pip install jax jaxlib --upgrade
```

Once JAX is installed, clone and install remaining requirements by running

```bash
git clone https://github.com/lionelmessi6410/ntga.git
cd ntga
pip install -r requirements.txt
```

If you only want to examine the effectiveness of NTGAs, you can download datasets [here](#unlearnable-datasets) and evaluate with `evaluate.py` or **any** code/model you prefer. To use `evaluate.py`, you do not need to install JAX externally, instead, all dependencies are specified in `requirements.txt`.

## Usage
### NTGA Attack
To generate poisoned data by NTGA, run

```bash
python generate_attack.py --model_type fnn --dataset cifar10 --save_path ./data/
```

There are few important arguments:
- `--model_type`: A string. Surrogate model used to craft poisoned data. One of `fnn` or `cnn`. `fnn` and `cnn` stands for the fully-connected and convolutional networks, respectively.
- `--dataset`: A string. One of `mnist`, `cifar10`, or `imagenet`. 
- `--t`: An integer. Time step used to craft poisoned data. Please refer to main paper for more details.
- `--eps`: A float. Strength of NTGA. The default settings for MNIST, CIFAR-10, and ImageNet are `0.3`, `8/255`, and `0.1`, respectively.
- `--nb_iter`: An integer. Number of iteration used to generate poisoned data.
- `--block_size`: An integer. Block size of B-NTGA algorithm.
- `--batch_size`: An integer.
- `--save_path`: A string.

In general, the attacks based on the FNN surrogate have greater influence against the fully-connected target networks, while the attacks based on the CNN surrogate work better against the convolutional target networks. Both `eps` and `block_size` influence the effectiveness of NTGA. Larger `eps` leads to stronger but more distinguishable perturbations, while larger `block_size` results in better collaborative effect (stronger attack) in NTGA but also induces both higher time and space complexities. If you encounter out-of-memory (OOM) errors, especially when using `--model_type cnn`, please try to reduce `block_size` and `batch_size` to save memory usage. The CNN surrogate takes more time and space to generate attacks, compared with the FNN surrogate under the same settings.

For ImageNet or another custom dataset, please specify the path to the dataset in the code directly. The original clean data and the poisoned ones crafted by NTGA can be founrd and downloaded in [Unlearnable Datasets](#unlearnable-datasets).

### Evaluation
Next, you can examine the effectiveness of the poisoned data crafted by NTGA by calling

```bash
python evaluate.py --model_type densenet121 --dataset cifar10 --dtype NTGA --x_train_path ./data/x_train_cifar10_ntga_cnn_best.npy --y_train_path ./data/y_train_cifar10.npy --batch_size 128 --save_path ./figure/
```

If you are interested in the performance on the clean data, run
```bash
python evaluate.py --model_type densenet121 --dataset cifar10 --dtype Clean --epoch 200 --batch_size 128 --save_path ./figures/
```

This code will also plot the learning curve and save it in `--save_path ./figures/`. The following figures show the results of [DenseNet121](https://arxiv.org/pdf/1608.06993.pdf) trained on the CIFAR-10 dataset. The left figure demonstrate the normal learning curve, where the network is trained on the clean data. On the contrary, the figure on the right-hand side shows the remarkable result of NTGA, where the training accuracy is \~100%, but the model fails to generalize.

<table border=0 >
	<tbody>
		<tr>
			<tr>
				<td width=50% > <img src="./figures/figure_cifar10_accuracy_densenet121_clean.png"> </td>
				<td width=50%> <img src="./figures/figure_cifar10_accuracy_densenet121_ntga.png"> </td>
			</tr>
		</tr>
	</tbody>
</table>

There are few important arguments:
- `--model_type`: A string. Target model used to evaluate poisoned data. One of `fnn`, `fnn_relu`, `cnn`, `resnet18`, `resnet34`, or `densenet121`.
- `--dataset`: A string. One of `mnist`, `cifar10`, or `imagenet`.
- `--dtype`: A string. One of `Clean` or `NTGA`, used for figure's title.
- `--x_train_path`: A string. Path for poisoned training data. Leave it empty for clean data (mnist or cifar10).
- `--y_train_path`: A string. Path for training labels. Leave it empty for clean data (mnist or cifar10).
- `--x_test_path`: A string. Path for testing data.
- `--y_test_path`: A string. Path for testing labels.
- `--epoch`: An integer.
- `--batch_size`: An integer.
- `--save_path`: A string.

### Visualization
How does the poisoned data look like? Is it really imperceptible to human? You can visualize the poisoned data and their normalized perturbations by calling

```bash
python plot_visualization.py --dataset cifar10 --x_train_path ./data/x_train_cifar10.npy --x_train_ntga_path ./data/x_train_cifar10_ntga_fnn_t1.npy --save_path ./figure/
```

The following figure shows some poisoned CIFAR-10 images. As we can see, they looks almost the same as the original clean data. However, training on the clean data can achieve \~92% test accuracy, while training on the poisoned data the performance decreases sharply to \~35%.
<p align="center">
	<img src="./figures/figure_cifar10_visualization.png" width=75%>
</p>

Here we also visualize the high-resolution ImageNet dataset and find even more interesting results:
<p align="center">
	<img src="./figures/figure_imagenet_visualization.png" width=75%>
</p>

The perturbations are nearly invisible! The only difference between the clean and poisoned images is the hue, which is easier to observe in the normalized perturbation.

There are few important arguments:
- `--dataset`: A string. One of `mnist`, `cifar10`, or `imagenet`.
- `--x_train_path`: A string. Path for clean training data. Leave it empty for clean data (mnist or cifar10).
- `--x_train_ntga_path`: A string. Path for poisoned training data. Leave it empty for clean data (mnist or cifar10).
- `--num`: An integer. Number of data to be visualized. The valid value is 1-5.")
- `--save_path`: A string.

## Results
Here we briefly report the performance of NTGA and some baselines. Please see the [main paper]() for a more complete results.
### FNN Surrogate
<table>
	<thead>
		<tr>
			<th align="left" width=20%>Target\Attack</th>
			<th align="center" width=16%>Clean</th>
			<td align="center" width=16%><a href="https://www.mdpi.com/2504-4990/1/1/11/htm" target="_blank"><b>RFA</b></a></td>
			<td align="center" width=16%><a href="https://proceedings.neurips.cc/paper/2019/file/1ce83e5d4135b07c0b82afffbe2b3436-Paper.pdf" target="_blank"><b>DeepConfuse</b></a></td>
			<th align="center" width=16%>NTGA(1)</th>
			<th align="center" width=16%>NTGA(best)</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<th align="left" colspan=6>Dataset: MNIST</th>
		</tr>
		<tr>
			<td align="left">FNN</td>
			<td align="center">96.26</td>
			<td align="center">74.23</td>
			<td align="center">-</td>
			<td align="center">3.95</td>
			<td align="center"><b>2.57</b></td>
		</tr>
		<tr>
			<td align="left">FNN-ReLU</td>
			<td align="center">97.87</td>
			<td align="center">84.62</td>
			<td align="center">-</td>
			<td align="center"><b>2.08</b></td>
			<td align="center">2.18</td>
		</tr>
		<tr>
			<td align="left">CNN</td>
			<td align="center">99.49</td>
			<td align="center">86.99</td>
			<td align="center">-</td>
			<td align="center">33.80</td>
			<td align="center"><b>26.03</b></td>
		</tr>
		<tr>
			<th align="left" colspan=6>Dataset: CIFAR-10</th>
		</tr>
		<tr>
			<td align="left">FNN</td>
			<td align="center">49.57</td>
			<td align="center">37.79</td>
			<td align="center">-</td>
			<td align="center">36.05</td>
			<td align="center"><b>20.63</b></td>
		</tr>
		<tr>
			<td align="left">FNN-ReLU</td>
			<td align="center">54.55</td>
			<td align="center">43.19</td>
			<td align="center">-</td>
			<td align="center">40.08</td>
			<td align="center"><b>25.95</b></td>
		</tr>
		<tr>
			<td align="left">CNN</td>
			<td align="center">78.12</td>
			<td align="center">74.71</td>
			<td align="center">-</td>
			<td align="center">48.46</td>
			<td align="center"><b>36.05</b></td>
		</tr>
		<tr>
			<td align="left">ResNet18</td>
			<td align="center">91.92</td>
			<td align="center">88.76</td>
			<td align="center">-</td>
			<td align="center">39.72</td>
			<td align="center"><b>39.68</b></td>
		</tr>
		<tr>
			<td align="left">DenseNet121</td>
			<td align="center">92.71</td>
			<td align="center">88.81</td>
			<td align="center">-</td>
			<td align="center"><b>46.50</b></td>
			<td align="center">47.36</td>
		</tr>
		<tr>
			<th align="left" colspan=6>Dataset: ImageNet</th>
		</tr>
		<tr>
			<td align="left">FNN</td>
			<td align="center">91.60</td>
			<td align="center">90.20</td>
			<td align="center">-</td>
			<td align="center"><b>76.60</b></td>
			<td align="center"><b>76.60</b></td>
		</tr>
		<tr>
			<td align="left">FNN-ReLU</td>
			<td align="center">92.20</td>
			<td align="center">89.60</td>
			<td align="center">-</td>
			<td align="center"><b>80.00</b></td>
			<td align="center"><b>80.00</b></td>
		</tr>
		<tr>
			<td align="left">CNN</td>
			<td align="center">96.00</td>
			<td align="center">95.80</td>
			<td align="center">-</td>
			<td align="center"><b>77.80</b></td>
			<td align="center"><b>77.80</b></td>
		</tr>
		<tr>
			<td align="left">ResNet18</td>
			<td align="center">99.80</td>
			<td align="center">98.20</td>
			<td align="center">-</td>
			<td align="center"><b>76.40</b></td>
			<td align="center"><b>76.40</b></td>
		</tr>
		<tr>
			<td align="left">DenseNet121</td>
			<td align="center">98.40</td>
			<td align="center">96.20</td>
			<td align="center">-</td>
			<td align="center"><b>72.80</b></td>
			<td align="center"><b>72.80</b></td>
		</tr>
	</tbody>
</table>

### CNN Surrogate
<table>
	<thead>
		<tr>
			<th align="left" width="20%">Target\Attack</th>
			<th align="center" width="16%">Clean</th>
			<td align="center" width="16%"><a href="https://www.mdpi.com/2504-4990/1/1/11/htm" target="_blank"><b>RFA</b></a></td>
			<td align="center" width="16%"><a href="https://proceedings.neurips.cc/paper/2019/file/1ce83e5d4135b07c0b82afffbe2b3436-Paper.pdf" target="_blank"><b>DeepConfuse</b></a></td>
			<th align="center" width="16%">NTGA(1)</th>
			<th align="center" width="16%">NTGA(best)</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<th align="left" colspan=6>Dataset: MNIST</th>
		</tr>
		<tr>
			<td align="left">FNN</td>
			<td align="center">96.26</td>
			<td align="center">69.95</td>
			<td align="center">15.48</td>
			<td align="center">8.46</td>
			<td align="center"><b>4.63</b></td>
		</tr>
		<tr>
			<td align="left">FNN-ReLU</td>
			<td align="center">97.87</td>
			<td align="center">84.15</td>
			<td align="center">17.50</td>
			<td align="center">3.48</td>
			<td align="center"><b>2.86</b></td>
		</tr>
		<tr>
			<td align="left">CNN</td>
			<td align="center">99.49</td>
			<td align="center">94.92</td>
			<td align="center">46.21</td>
			<td align="center">23.89</td>
			<td align="center"><b>15.64</b></td>
		</tr>
		<tr>
			<th align="left" colspan=6>Dataset: CIFAR-10</th>
		</tr>
		<tr>
			<td align="left">FNN</td>
			<td align="center">49.57</td>
			<td align="center">41.31</td>
			<td align="center">32.59</td>
			<td align="center">28.84</td>
			<td align="center"><b>28.81</b></td>
		</tr>
		<tr>
			<td align="left">FNN-ReLU</td>
			<td align="center">54.55</td>
			<td align="center">46.87</td>
			<td align="center">35.06</td>
			<td align="center">32.77</td>
			<td align="center"><b>32.11</b></td>
		</tr>
		<tr>
			<td align="left">CNN</td>
			<td align="center">78.12</td>
			<td align="center">73.80</td>
			<td align="center">44.84</td>
			<td align="center">41.17</td>
			<td align="center"><b>40.52</b></td>
		</tr>
		<tr>
			<td align="left">ResNet18</td>
			<td align="center">91.92</td>
			<td align="center">89.54</td>
			<td align="center">41.10</td>
			<td align="center">34.74</td>
			<td align="center"><b>33.29</b></td>
		</tr>
		<tr>
			<td align="left">DenseNet121</td>
			<td align="center">92.71</td>
			<td align="center">90.50</td>
			<td align="center">54.99</td>
			<td align="center">43.54</td>
			<td align="center"><b>37.79</b></td>
		</tr>
		<tr>
			<th align="left" colspan=6>Dataset: ImageNet</th>
		</tr>
		<tr>
			<td align="left">FNN</td>
			<td align="center">91.60</td>
			<td align="center">87.80</td>
			<td align="center">90.80</td>
			<td align="center"><b>75.80</b></td>
			<td align="center"><b>75.80</b></td>
		</tr>
		<tr>
			<td align="left">FNN-ReLU</td>
			<td align="center">92.20</td>
			<td align="center">87.60</td>
			<td align="center">91.00</td>
			<td align="center"><b>80.00</b></td>
			<td align="center"><b>80.00</b></td>
		</tr>
		<tr>
			<td align="left">CNN</td>
			<td align="center">96.00</td>
			<td align="center">94.40</td>
			<td align="center">93.00</td>
			<td align="center"><b>79.00</b></td>
			<td align="center"><b>79.00</b></td>
		</tr>
		<tr>
			<td align="left">ResNet18</td>
			<td align="center">99.80</td>
			<td align="center">96.00</td>
			<td align="center">92.80</td>
			<td align="center"><b>76.40</b></td>
			<td align="center"><b>76.40</b></td>
		</tr>
		<tr>
			<td align="left">DenseNet121</td>
			<td align="center">98.40</td>
			<td align="center">90.40</td>
			<td align="center">92.80</td>
			<td align="center"><b>80.60</b></td>
			<td align="center"><b>80.60</b></td>
		</tr>
	</tbody>
</table>

## Unlearnable Datasets

<!-- MNIST FNN 64
MNIST CNN 64

CIFAR FNN 4096
CIFAR CNN 8

IMAGENET FNN 1
IMAGENET CNN 1 -->

<!-- ## Citation
If you find this code is helpful for your research, please cite our [ICML 2021 paper]():
```
@inproceedings{wu2020adversarial,
	title={Adversarial Robustness via Runtime Masking and Cleansing},
	author={Wu, Yi-Hsuan and Yuan, Chia-Hung and Wu, Shan-Hung},
	booktitle={International Conference on Machine Learning},
	pages={10399--10409},
	year={2020},
	organization={PMLR}
}
``` -->
