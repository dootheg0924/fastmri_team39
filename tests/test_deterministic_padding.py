import pytest
import torch
import torch.nn.functional as F

from utils.model.unet import deterministic_reflect_pad2d


@pytest.mark.parametrize(
    ("shape", "padding"),
    [
        ((1, 2, 5, 7), (0, 1, 0, 1)),
        ((2, 3, 6, 9), (2, 3, 1, 4)),
        ((1, 1, 4, 5), (1, 0, 2, 0)),
        ((1, 1, 3, 3), (0, 0, 0, 0)),
    ],
)
def test_deterministic_reflect_pad_matches_pytorch(shape, padding):
    image = torch.randn(shape)

    expected = F.pad(image, padding, mode="reflect")
    actual = deterministic_reflect_pad2d(image, padding)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deterministic_reflect_pad_matches_pytorch_gradient():
    image = torch.randn(2, 3, 6, 9, dtype=torch.float64)
    native_input = image.clone().requires_grad_()
    deterministic_input = image.clone().requires_grad_()
    padding = (2, 3, 1, 4)
    weights = torch.randn(2, 3, 11, 14, dtype=torch.float64)

    (F.pad(native_input, padding, mode="reflect") * weights).sum().backward()
    (
        deterministic_reflect_pad2d(deterministic_input, padding) * weights
    ).sum().backward()

    torch.testing.assert_close(
        deterministic_input.grad,
        native_input.grad,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_deterministic_reflect_pad_cuda_backward_is_allowed():
    was_enabled = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        image = torch.randn(
            1,
            2,
            5,
            7,
            device="cuda",
            requires_grad=True,
        )
        deterministic_reflect_pad2d(image, (2, 1, 3, 0)).square().sum().backward()
    finally:
        torch.use_deterministic_algorithms(was_enabled)
