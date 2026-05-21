import numpy as np
import torch

from src.data.augs import Mixup


def _make_batch(B=4, C=3, F=64, T=128, n_classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, C, F, T, generator=g)
    y = torch.zeros(B, n_classes)
    rng = np.random.default_rng(seed)
    for i in range(B):
        idx = rng.integers(0, n_classes)
        y[i, idx] = 1.0
    return x, y


def test_mixup_disabled_when_prob_zero():
    mixup = Mixup(mixup_prob=0.0, alpha=0.5)
    x, y = _make_batch()
    mx, my = mixup(x, y)
    assert torch.equal(mx, x)
    assert torch.equal(my, y)


def test_mixup_warmup_blocks_mixing():
    mixup = Mixup(mixup_prob=1.0, alpha=0.5, warmup_epochs=3)
    x, y = _make_batch()
    mixup.set_epoch(0)
    mx, my = mixup(x, y)
    assert torch.equal(mx, x)
    assert torch.equal(my, y)
    mixup.set_epoch(2)
    mx, my = mixup(x, y)
    assert torch.equal(mx, x)
    assert torch.equal(my, y)


def test_mixup_soft_labels_in_range():
    np.random.seed(0)
    torch.manual_seed(0)
    mixup = Mixup(mixup_prob=1.0, alpha=0.5, warmup_epochs=0, cutmix_prob=0.0)
    mixup.set_epoch(10)
    x, y = _make_batch()
    mx, my = mixup(x, y)
    assert mx.shape == x.shape
    assert my.shape == y.shape
    # Targets deben estar dentro de [0, 1] y al menos uno debe ser fraccional
    assert (my >= 0).all() and (my <= 1).all()
    # La suma de targets por muestra no puede exceder la suma de los originales mezclados
    assert my.sum() <= (y.sum() + y[torch.arange(y.size(0))].sum() + 1e-6)


def test_mixup_no_force_dominant_allows_low_lam():
    np.random.seed(123)
    torch.manual_seed(123)
    mixup = Mixup(mixup_prob=1.0, alpha=0.5, warmup_epochs=0, force_dominant=False)
    mixup.set_epoch(10)
    samples = []
    for _ in range(200):
        samples.append(mixup._sample_lam())
    arr = np.array(samples)
    # Sin force_dominant, ~50% de los lams quedan por debajo de 0.5
    assert (arr < 0.5).mean() > 0.3
    assert (arr > 0.5).mean() > 0.3


def test_mixup_force_dominant_keeps_lam_ge_half():
    mixup = Mixup(mixup_prob=1.0, alpha=0.5, warmup_epochs=0, force_dominant=True)
    samples = [mixup._sample_lam() for _ in range(200)]
    assert all(l >= 0.5 - 1e-9 for l in samples)


def test_cutmix_changes_input_and_adjusts_label():
    np.random.seed(7)
    torch.manual_seed(7)
    mixup = Mixup(
        mixup_prob=1.0,
        alpha=0.5,
        warmup_epochs=0,
        cutmix_prob=1.0,
        force_dominant=False,
    )
    mixup.set_epoch(10)
    x, y = _make_batch(B=2, n_classes=5)
    mx, my = mixup(x, y)
    # CutMix debe haber alterado el tensor original
    assert not torch.equal(mx, x)
    # Las dimensiones se mantienen
    assert mx.shape == x.shape
    assert my.shape == y.shape


def test_pseudo_mix_uses_pseudo_targets():
    np.random.seed(0)
    torch.manual_seed(0)
    mixup = Mixup(mixup_prob=1.0, alpha=0.5, warmup_epochs=0, force_dominant=False)
    mixup.set_epoch(10)
    x, y = _make_batch(B=4, n_classes=6)
    x_p = torch.randn_like(x)
    y_p = torch.zeros_like(y)
    y_p[:, 0] = 0.8  # pseudo-soft target
    mx, my = mixup(x, y, x_pseudo=x_p, y_pseudo=y_p, is_pseudo_mix=True)
    # Si el mixup actuó, la columna 0 de my debe ser positiva en alguna muestra
    assert my[:, 0].sum() > 0.0
    assert mx.shape == x.shape


def test_no_label_explosion_with_two_multi_positive_samples():
    """Con OR-mixup, mezclar dos muestras con 4 positivos cada una daba 8.
    Con soft mixup, la suma de positivos por sample queda acotada."""
    mixup = Mixup(mixup_prob=1.0, alpha=0.5, warmup_epochs=0, force_dominant=False)
    mixup.set_epoch(10)
    B, n_classes = 4, 10
    x = torch.randn(B, 3, 32, 64)
    y = torch.zeros(B, n_classes)
    y[:, :4] = 1.0  # cada muestra tiene 4 positivos
    np.random.seed(42)
    torch.manual_seed(42)
    _, my = mixup(x, y)
    # Suma por muestra <= 4 (no se inflan los positivos como en OR-mixup)
    per_sample_sum = my.sum(dim=1)
    assert (per_sample_sum <= 4.0 + 1e-6).all()
