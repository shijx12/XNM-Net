# eXplainable and eXplicit Neural Modules (XNMs)

Pytorch implementation of CVPR 2019 paper 

**[Explainable and Explicit Visual Reasoning over Scene Graphs](https://arxiv.org/abs/1812.01855)**
<br>
[Jiaxin Shi](https://shijx12.github.io), [Hanwang Zhang](http://www.ntu.edu.sg/home/hanwangzhang/#aboutme), Juanzi Li

Flowchart:
![flowchart](images/flowchart.pdf)

<br>
A visualization of our reasoning process:

![example](images/example.pdf)


**Abstract**
We aim to dismantle the prevalent black-box neural architectures used in complex visual reasoning tasks, into the proposed eXplainable and eXplicit Neural Modules (XNMs), which advance beyond existing neural module networks towards using scene graphs --- objects as nodes and the pairwise relationships as edges --- for explainable and explicit reasoning with structured knowledge. XNMs allow us to pay more attention to teach machines how to ''think'', regardless of what they ''look''. As we will show in the paper, by using scene graphs as an inductive bias, 1) we can design XNMs in a concise and flexible fashion, i.e., XNMs merely consist of 4 meta-types, which significantly reduce the number of parameters by 10 to 100 times, and 2) we can explicitly trace the reasoning-flow in terms of graph attentions. XNMs are so generic that they support a wide range of scene graph implementations with various qualities. For example, when the graphs are detected perfectly, XNMs achieve 100% accuracy on both CLEVR and CLEVR CoGenT, establishing an empirical performance upper-bound for visual reasoning; when the graphs are noisily detected from real-world images, XNMs are still robust to achieve a competitive 67.5% accuracy on VQAv2.0, surpassing the popular bag-of-objects attention models without graph structures. 

If you find this code useful in your research, please cite
``` tex
@article{shi2018explainable,
    title={Explainable and Explicit Visual Reasoning over Scene Graphs},
    author={Shi, Jiaxin and Zhang, Hanwang and Li, Juanzi},
    journal={arXiv preprint arXiv:1812.01855},
    year={2018}
}
```



## Requirements
- python==3.6
- pytorch==0.4.0
- h5py 
- tqdm
- matplotlib


## Experiments
We have 4 experiment settings:
- CLEVR dataset, Det setting (i.e., using detected scene graphs). Codes are in the directory `./exp_clevr_detected`.
- CLEVR dataset, GT setting (i.e., using ground truth scene graphs), attention is computed by softmax function over the label space. Codes are in `./exp_clevr_gt_softmax`.
- CLEVR dataset, GT setting, attention is computed by sigmoid function. Codes are in `./exp_clevr_gt_sigmoid`.
- VQA2.0 dataset, detected scene graphs. Codes are in `./exp_vqa`.

We have a **separate README for each experiment setting** as an instruction to reimplement our reported results.
Feel free to contact me if you have any problems: shijx12@gmail.com

## Acknowledgement
- We refer to the repo [clevr-iep](https://github.com/facebookresearch/clevr-iep) for preprocessing codes.
- Our implementations of module model and dataloader are based on [tbd-net](https://github.com/davidmascharka/tbd-nets).
- Our stacked neural module implementation in `./exp_vqa` is based on Hu's [StackNMN](https://github.com/ronghanghu/snmn).

