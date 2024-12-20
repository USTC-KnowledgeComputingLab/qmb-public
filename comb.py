import torch

pc = []
pp = []

for i in range(16):
    configs, psi, site, kind, coef = torch.load(f"C{i}.pth", weights_only=True)
    pc.append(configs)
    pp.append(psi)

configs = torch.cat(pc)
psi = torch.cat(pp)

torch.save((configs, psi, site, kind, coef), f"comb.pth")
