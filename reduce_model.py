import click
import torch


@click.command
@click.option("--ckpt")
def reduce_ckpt(ckpt):
    # 加载原始检查点
    checkpoint = torch.load(ckpt, map_location="cpu")
    print(checkpoint.keys())
    # 只保留 state_dict
    new_checkpoint = {"state_dict": checkpoint["state_dict"], "hyper_parameters": checkpoint["hyper_parameters"]}

    # 保存新的精简检查点
    torch.save(new_checkpoint, ckpt.replace(".ckpt", "_reduced.ckpt"))


if __name__ == '__main__':
    reduce_ckpt()
