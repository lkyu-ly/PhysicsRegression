#!/usr/bin/env python3
import numpy as np
import torch
import paddle


def main():
    rng = np.random.default_rng(0)
    logits_np = rng.normal(size=(37, 11)).astype("float32")
    labels_np = rng.integers(0, 11, size=(37,), dtype=np.int64)

    torch_logits = torch.tensor(logits_np)
    torch_labels = torch.tensor(labels_np, dtype=torch.long)
    torch_loss = torch.nn.functional.cross_entropy(torch_logits.float(), torch_labels, reduction="mean")

    paddle_logits = paddle.to_tensor(logits_np)
    paddle_labels = paddle.to_tensor(labels_np, dtype="int64")
    paddle_loss = paddle.nn.functional.cross_entropy(
        input=paddle_logits.astype("float32"),
        label=paddle_labels,
        reduction="mean",
    )

    abs_err = abs(float(torch_loss.item()) - float(paddle_loss.item()))
    print(f"torch_loss={torch_loss.item():.12f}")
    print(f"paddle_loss={paddle_loss.item():.12f}")
    print(f"abs_err={abs_err:.12e}")
    assert abs_err < 1e-6, abs_err


if __name__ == "__main__":
    main()
