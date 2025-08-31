import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("compiled archs:", torch.cuda.get_arch_list())  # 여기 목록에 'sm_120'이 보여야 정상

# 커널 테스트
x = torch.randn(1024,1024, device="cuda")
y = x @ x.T
print(float(y.sum()))