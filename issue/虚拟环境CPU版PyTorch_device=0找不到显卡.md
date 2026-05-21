##### 卸载当前 CPU 版 PyTorch：

```shell
python -m pip uninstall -y torch torchvision torchaudio	
```

##### 升级 pip：

```shell
python -m pip install --upgrade pip
```

##### 安装 CUDA 版 PyTorch，优先用 CUDA 12.8：

```shell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

##### 安装完验证

```sh
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('device count:', torch.cuda.device_count()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

正常结果应该类似：

```sh
torch: 2.x.x+cu128
cuda: True
cuda version: 12.8
device count: 1
gpu: NVIDIA RTX 500 Ada Generation
```