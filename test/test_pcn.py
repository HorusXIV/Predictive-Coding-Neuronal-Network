"""Regression checks for the PCN. Run: python test_pcn.py (or pytest)."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pcn.model import NEGATIVE_SLOPE, PredictiveCodingNetwork, leaky_relu_deriv


def _toy_classification(n=400, in_dim=6, num_classes=3, seed=0):
    """Linearly separable toy problem: y = argmax(x @ w)."""
    torch.manual_seed(seed)
    w = torch.randn(in_dim, num_classes)
    x = torch.randn(n, in_dim)
    return x, (x @ w).argmax(dim=1)


def test_activation_deriv_matches_autograd():
    a = torch.randn(200, requires_grad=True)
    expected, = torch.autograd.grad(F.leaky_relu(a, NEGATIVE_SLOPE).sum(), a)
    assert torch.allclose(expected, leaky_relu_deriv(a.detach()))


def test_latent_init_is_deterministic_and_depends_on_input():
    """The original bug: latents were randn, so x_L carried no input signal."""
    model = PredictiveCodingNetwork(dims=[8, 6, 4], output_dim=3)
    x = torch.randn(16, 8)
    assert all(torch.equal(p, q) for p, q in zip(model.init_latents(x), model.init_latents(x)))
    assert not torch.allclose(model.init_latents(x)[-1], model.init_latents(torch.zeros_like(x))[-1])


def test_prediction_is_reproducible_for_the_same_input():
    """Two eval passes used to agree on ~50% of samples -- i.e. pure guessing."""
    model = PredictiveCodingNetwork(dims=[8, 6, 4], output_dim=3)
    x = torch.randn(32, 8)
    a = model.run(mode="predictive", x=x, T_infer=20, device="cpu").output_prediction
    b = model.run(mode="predictive", x=x, T_infer=20, device="cpu").output_prediction
    assert torch.allclose(a, b)


def test_generative_mode_does_not_mutate_caller_tensor():
    model = PredictiveCodingNetwork(dims=[8, 6, 4], output_dim=3)
    x = torch.randn(3, 8)
    before = x.clone()
    model.run(mode="generative", y=torch.tensor([0, 1, 2]), x0_init=x, T_infer=5, device="cpu")
    assert torch.equal(x, before)


def test_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    from pcn.model import _f1_from_confusion

    torch.manual_seed(0)
    for num_classes, n in [(2, 500), (3, 60), (10, 2000)]:
        true = torch.randint(0, num_classes, (n,))
        pred = torch.randint(0, num_classes, (n,))
        confusion = torch.bincount(
            true * num_classes + pred, minlength=num_classes**2
        ).reshape(num_classes, num_classes)
        got = _f1_from_confusion(confusion)
        assert abs(got["f1_macro"] - f1_score(true, pred, average="macro", zero_division=0)) < 1e-6
        if num_classes == 2:
            assert abs(got["f1"] - f1_score(true, pred, zero_division=0)) < 1e-6

    # a class absent from both labels and predictions is dropped, not scored 0
    absent = torch.bincount(torch.zeros(20, dtype=torch.long), minlength=9).reshape(3, 3)
    assert _f1_from_confusion(absent)["f1_macro"] == 1.0


def test_default_optimizer_reproduces_the_manual_update():
    """SGD(lr, weight_decay) must equal ``W -= lr * (grad + wd * W)`` exactly."""
    torch.manual_seed(0)
    x, y = _toy_classification(n=64)
    y_target = F.one_hot(y[:32], num_classes=3).float()
    eta_learn, weight_decay = 0.01, 0.05

    torch.manual_seed(1)
    model = PredictiveCodingNetwork(dims=[6, 10, 8], output_dim=3)
    latents = [x[:32]] + model.init_latents(x[:32])
    with torch.no_grad():
        for _ in range(10):
            model.predictive_step(latents, eta_infer=0.1, output_target=y_target)

        # manual rule, applied to a copy of the weights
        model.set_weight_gradients(latents, y_target)
        manual = [
            (p - eta_learn * (p.grad + weight_decay * p)).clone()
            for p in model.parameters()
        ]

        # optimizer path
        optimizer = torch.optim.SGD(
            model.parameters(), lr=eta_learn, weight_decay=weight_decay
        )
        optimizer.zero_grad(set_to_none=True)
        model.set_weight_gradients(latents, y_target)
        optimizer.step()

    for expected, param in zip(manual, model.parameters()):
        assert torch.allclose(expected, param), (expected - param).abs().max()


def test_fit_accepts_a_dataset_and_honours_batch_size():
    x, y = _toy_classification(n=128)
    model = PredictiveCodingNetwork(dims=[6, 10, 8], output_dim=3)
    history = model.fit(
        TensorDataset(x, y),
        num_epochs=1,
        eta_infer=0.1,
        T_infer=5,
        batch_size=16,
        device="cpu",
        eval_every=1,
    )
    assert history["config"]["batch_size"] == 16
    assert history["config"]["optimizer"] == "SGD"

    # an existing DataLoader is passed through untouched
    loader = DataLoader(TensorDataset(x, y), batch_size=64)
    history = model.fit(
        loader, num_epochs=1, eta_infer=0.1, T_infer=5, batch_size=999,
        device="cpu", record_history=False,
    )
    assert history["config"]["batch_size"] == 64


def test_fit_accepts_a_custom_optimizer():
    x, y = _toy_classification(n=128)
    model = PredictiveCodingNetwork(dims=[6, 10, 8], output_dim=3)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = model.fit(
        TensorDataset(x, y),
        num_epochs=2,
        eta_infer=0.1,
        T_infer=5,
        optimizer=optimizer,
        device="cpu",
        record_history=False,
    )
    assert history["config"]["optimizer"] == "Adam"
    assert any(not torch.allclose(b, p) for b, p in zip(before, model.parameters()))


def test_evaluate_reports_f1_and_does_not_leave_training_mode_changed():
    x, y = _toy_classification(n=64)
    model = PredictiveCodingNetwork(dims=[6, 10, 8], output_dim=3)
    model.train()
    metrics = model.evaluate(TensorDataset(x, y), eta_infer=0.1, T_infer=5, device="cpu")
    assert model.training, "evaluate() must restore the previous mode"
    assert {"accuracy", "f1_macro", "free_energy"} <= metrics.keys()
    assert metrics["n"] == 64


def test_learns_a_linearly_separable_task():
    """End-to-end: accuracy must beat chance. Catches a silently dead readout."""
    x, y = _toy_classification(n=400)
    torch.manual_seed(0)
    model = PredictiveCodingNetwork(dims=[6, 16, 12], output_dim=3)
    model.fit(
        TensorDataset(x, y),
        num_epochs=60,
        eta_infer=0.1,
        T_infer=50,
        batch_size=32,
        eta_learn=0.01,
        device="cpu",
        record_history=False,
    )
    accuracy = model.evaluate(
        TensorDataset(x, y), eta_infer=0.1, T_infer=50, device="cpu"
    )["accuracy"]
    assert accuracy > 0.7, f"chance is ~0.33, got {accuracy:.3f}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
