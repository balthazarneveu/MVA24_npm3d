from pointnet import Tnet, MLP, PointNetBasic, PointNetFull
import torch
import pytest


def test_tnet():
    # Check TNet inference and that the Tnet is permutation invariant
    with torch.no_grad():
        model = Tnet()
        model.eval()  # EVALUATION MODE to disable dropout!
        inp = torch.rand(32, 3, 1024)
        out, mat = model(inp)
        out_p, mat_p = model(inp[:, :, torch.randperm(1024)])
        assert torch.allclose(mat, mat_p)
        assert mat.shape == (32, 3, 3)


def test_mlp():
    model = MLP(10)
    out, mat = model(torch.rand(32, 3, 1024))
    assert mat is None
    assert out.shape == (32, 10)


@pytest.mark.parametrize("model_name", ["mlp", "pointnet", "pointnet_tnet"])
def test_permutation_invariance(model_name):
    with torch.no_grad():
        if model_name == "pointnet":
            model = PointNetBasic()
            model.eval()  # EVALUATION MODE to disable dropout!
        if model_name == "pointnet_tnet":
            model = PointNetFull()
        if model_name == "mlp":
            model = MLP(10)
        model.eval()  # EVALUATION MODE to disable dropout!
        inp = torch.rand(32, 3, 1024)
        out_1, mat = model(inp)
        out_2, mat2 = model(inp[:, :, torch.randperm(1024)])
        if model_name == "pointnet" or model_name == "pointnet_tnet":
            assert torch.allclose(out_1, out_2)
            if model_name == "pointnet_tnet":
                # Redundant check on Tnet permutation invariant
                assert torch.allclose(mat, mat2)
        if model_name == "mlp":
            assert mat is None
            assert not torch.allclose(out_1, out_2)
