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
sbatch ml-loads/simple_inference_slurm.sh
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