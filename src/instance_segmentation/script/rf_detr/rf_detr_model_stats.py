#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script provides a detailed breakdown of the RF-DETR model's parameters, including total parameters,
memory size in FP32 and FP16 formats, and a per-module analysis.
It also lists the largest parameter tensors and provides a breakdown of the backbone and transformer components of the model.


@author: Olegs
"""

# ==========================================================
from rfdetr import RFDETRNano

# ------------------------------------------------------------
# RF-DETR model statistics
# Assumes:
#   model = RFDETRNano(...)
# ------------------------------------------------------------


model = RFDETRNano(resolution=384,
                   pretrain_weights="EXPERIMENT_PATH/checkpoint_best_total.pth")


root = model.model.model   # RF-DETR network

# ---------- Helper functions ----------

def num_params(module):
    return sum(p.numel() for p in module.parameters())

def num_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def fp32_size_mb(module):
    return num_params(module) * 4 / 1024**2

def fp16_size_mb(module):
    return num_params(module) * 2 / 1024**2

def tensor_count(module):
    return len(list(module.parameters()))

# ---------- Total model ----------

total_params = num_params(root)

print("=" * 90)
print("RF-DETR MODEL STATISTICS")
print("=" * 90)
print(f"Total parameters : {total_params:,}")
print(f"FP32 size        : {total_params * 4 / 1024**2:.2f} MB")
print(f"FP16 size        : {total_params * 2 / 1024**2:.2f} MB")
print()

# ---------- Per top-level module ----------

header = (
    f"{'Module':20s}"
    f"{'Params(M)':>12s}"
    f"{'Train(M)':>12s}"
    f"{'FP32(MB)':>12s}"
    f"{'FP16(MB)':>12s}"
    f"{'%Total':>10s}"
    f"{'Tensors':>10s}"
)

print(header)
print("-" * len(header))

for name, module in root.named_children():

    p = num_params(module)
    pt = num_trainable(module)

    print(
        f"{name:20s}"
        f"{p/1e6:12.3f}"
        f"{pt/1e6:12.3f}"
        f"{fp32_size_mb(module):12.2f}"
        f"{fp16_size_mb(module):12.2f}"
        f"{100*p/total_params:10.2f}"
        f"{tensor_count(module):10d}"
    )

print()

# ---------- Largest parameter tensors ----------

print("=" * 90)
print("Largest parameter tensors")
print("=" * 90)

params = []

for name, param in root.named_parameters():
    params.append(
        (
            name,
            param.numel(),
            tuple(param.shape),
            param.numel() * 4 / 1024**2,
        )
    )

params.sort(key=lambda x: x[1], reverse=True)

for name, n, shape, mb in params[:20]:
    print(f"{mb:8.2f} MB   {n:10,d}   {str(shape):30s}   {name}")

print()

# ---------- Backbone details ----------

print("=" * 90)
print("Backbone breakdown")
print("=" * 90)

backbone = root.backbone

for name, module in backbone.named_children():
    p = num_params(module)
    if p == 0:
        continue

    print(
        f"{name:30s}"
        f"{p/1e6:10.3f} M"
        f"{fp32_size_mb(module):10.2f} MB"
    )

print()

# ---------- Transformer details ----------

print("=" * 90)
print("Transformer breakdown")
print("=" * 90)

transformer = root.transformer

for name, module in transformer.named_children():
    p = num_params(module)
    if p == 0:
        continue

    print(
        f"{name:30s}"
        f"{p/1e6:10.3f} M"
        f"{fp32_size_mb(module):10.2f} MB"
    )


