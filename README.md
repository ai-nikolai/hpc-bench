# HPC Bench

A small repo for various HPC tasks... Maybe one day this will turn into a proper HPC bench... For now we want to have useful commands / function / evaluation scripts that will help us evaluate the HPC envs properly....


---

## VM:

### Some generally useful commands:
```bash
export HF_HOME="/workspace"
# export PYPI_HOME=
```

---
## CSCC Cluster:
Partition:
- cscc-gpu-p

QoS:
- cscc-gpu-qos

```bash
#! /bin/bash
#SBATCH --job-name=test_job # Job name
#SBATCH --output=/home/username/output_.%A.txt 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=40G 
#SBATCH --cpus-per-task=64 
#SBATCH --gres=gpu:1 
#SBATCH -p cscc-gpu-p 
#SBATCH --time=12:00:00 
#SBATCH -q cscc-gpu-qos
```




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

#### Attempt 1: vllm. Installation guide from scratch of vllm (Tried: 26.08.2025) -> Result: Failed (see point 6...)
1. Create virtual env:
```bash
python3 -m virtualenv env_llm
source env_llm/bin/activate #activates env.
```

2. Install pytorch: (this is probably too new...)
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

3. Installing the (custom) rocm-build dependencies...(however, I am using the newer pytorhc) let's see (also, from the official docs I am skipping the explicit triton installation, as the hope is that it works out of the box now...)
```bash
cd vllm/requirements/
# pip3 install -r rocm-build.txt #taking dependencies from there
echo """
-r common.txt

triton==3.3.0
cmake>=3.26.1,<4
packaging>=24.2
setuptools>=77.0.3,<80.0.0
setuptools-scm>=8
wheel
jinja2>=3.1.6
amdsmi==6.2.4""" > rocm_build_test.txt

pip3 install -r rocm_build_test.txt
```

4. Installing rocm dependencies:
```bash
pip3 install -r rocm.txt
```

5. Building vllm itself: (Again following documentation from step: 3, kind of...)
```bash
cd ..
# MI210 or MI250 = gfx90a; MI300 = gfx942. Could also do: PYTORCH_ROCM_ARCH="gfx90a;gfx942"
export PYTORCH_ROCM_ARCH="gfx90a"
python3 setup.py develop
```

6. Lol, Results in error: (will need to retry the above with the correct pytorch / rocm version...)
```log
AttributeError: /opt/rocm-6.3.3/lib/libamd_smi.so: undefined symbol: amdsmi_reset_gpu_compute_partition. Did you mean: 'amdsmi_set_gpu_compute_partition'?
```



### Getting SGLang running:

#### Attempt 1: sglang. Installation from scratch of SGLang (Tried: 25.08.2025) -> Result: did not work, needs vllm installation... Will attempt vllm installation & then sglang.

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