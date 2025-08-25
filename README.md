# HPC Bench

A small repo for various HPC tasks... Maybe one day this will turn into a proper HPC bench... For now we want to have useful commands / function / evaluation scripts that will help us evaluate the HPC envs properly....


---

## CAMD Cluster:
1. Getting vllm image:
```bash
apptainer build ./apptainer_images/vllm-rocm.sif docker://rocm/vllm:latest
```

2. Running inference using vllm.
```bash
sbatch ml_loads/slurm_vllm.sh
```

3. Running inference using sglang.
```bash
sbatch ml_loads/slurm_sglang.sh
```


### Getting vllm running.

1. Approach 1 apptainer is working if you specific the right `-x ` options (to not interfere with the other servers.)

2. Approach 2, direct VLLM installation... [official docs](https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html#build-from-source-rocm)

### Getting SGLang running:

#### Installation from scratch of SGLang (Tried: 25.08.2025) -> Result: did not work, needs vllm installation... Will attempt vllm installation & then sglang.

1. Create virtual env:
```bash
python3 -m virtualenv env_llm
source env_llm/bin/activate #activates env.
```
2. Install pytorch:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

3. Install SGlang [official docs](https://docs.sglang.ai/platforms/amd_gpu.html)
```bash
# Use the last release branch
git clone -b v0.5.1.post2 https://github.com/sgl-project/sglang.git

# Compile sgl-kernel
cd sglang/sgl-kernel
python setup_rocm.py install

# Install sglang python package
cd ..
pip install -e "python[all_hip]"
```

## FAQ - Some common issues:

### Configuring Proxy for Github
```bash
git config --global http.proxy $http_proxy
git config --global https.proxy $https_proxy
git config --global http.proxyAuthMethod 'basic'
```

```bash
git config --global user.name "Nikolai Rozanov"
git config --global user.email "nikolai.rozanov@gmail.com"
```