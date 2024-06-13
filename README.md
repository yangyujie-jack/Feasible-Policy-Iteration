# Feasible Policy Iteration

Code of the paper "Feasible Policy Iteration".\
[paper](https://arxiv.org/abs/2304.08845)

## Installation

```bash
mamba create -n fpi python=3.10
mamba activate fpi

pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install safety-gymnasium

git clone git@git.tsinghua.edu.cn:yangyj21/fpi.git
cd fpi
pip install -e .
```
