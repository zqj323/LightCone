# LightCone
# LightCone: 高鲁棒性抽象推理模型
A High-Robustness Abstract Reasoning Model for Noise-Resilient Abstract Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![CNKI](https://img.shields.io/badge/Paper-CNKI-green.svg)](https://www.cnki.net/)

## 简介
本项目实现 **LightCone** 高鲁棒性抽象推理模型，针对噪声干扰场景设计，包含**动态阈值调节、边界软约束、自洽校验**三大创新机制，显著提升模型在复杂环境下的推理稳定性。

可应用于：
- ARC 抽象推理任务
- 政务涉密数据研判
- 工业异常检测
- 逻辑推理与边缘计算场景

## 创新点
- 推理网络 RN + 逻辑分类头 LC 主干结构
- 动态阈值自适应模块（训练阶段 + 噪声强度）
- 边界软约束机制，平衡泛化与稳定性
- 自洽校验与反馈优化，提升结果可靠性
- 联合损失函数 + Adam+Lookahead 优化策略

## 实验性能
- 干净数据：**92.28%** 准确率
- 噪声 0.05~0.20：较 WideResNet 提升 **2.48%~4.37%**
- 噪声强度越大，优势越显著

## 环境安装
```bash
pip install -r requirements.txt
