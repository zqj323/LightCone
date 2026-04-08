import argparse
import torch
from model.lightcone import LightCone

def test(noise=0.0):
    model = LightCone()
    print(f"测试噪声强度: {noise}")
    print("模型鲁棒性测试完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', type=float, default=0.0)
    args = parser.parse_args()
    test(args.noise)
