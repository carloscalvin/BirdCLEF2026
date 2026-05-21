import numpy as np
import torch
import torch.nn as nn


class Mixup(nn.Module):
    """Mixup + CutMix temporal con soft labels para entrenamiento SED multi-etiqueta.

    Cambios clave respecto a la versión OR-mixup anterior:
      * Soft labels: ``mixed_y = lam * y_a + (1 - lam) * y_b`` (compatible con BCE).
        Esto elimina el desajuste anterior entre input mezclado de forma suave y
        target mezclado en modo OR (que producía falsos positivos y mala
        calibración al pedir target=1.0 con señal al 5-10%).
      * Sin ``lam = max(lam, 1 - lam)`` por defecto (configurable). Mezclar y
        además forzar al input a ser dominante mientras se pide etiqueta plena
        de ambas clases es exactamente la fuente del problema.
      * CutMix temporal opcional: copia un segmento del eje tiempo del segundo
        sample, ajustando ``lam`` por la fracción cortada. Para SED esto es más
        natural que el blending alpha porque preserva el espectro original.
      * Warmup por epoch (``set_epoch``): los primeros epochs se desactiva el
        mixup para que el modelo aprenda representaciones estables antes de
        intentar disociar inputs mezclados.
      * Soporte para pseudo-labels en modo soft: si ``y_pseudo`` viene con
        probabilidades (no binarizadas), se mezclan tal cual.
    """

    def __init__(
        self,
        mixup_prob: float = 0.5,
        alpha: float = 0.5,
        cutmix_prob: float = 0.0,
        warmup_epochs: int = 0,
        force_dominant: bool = False,
    ):
        super().__init__()
        self.mixup_prob = float(mixup_prob)
        self.alpha = float(alpha)
        self.cutmix_prob = float(cutmix_prob)
        self.warmup_epochs = int(warmup_epochs)
        self.force_dominant = bool(force_dominant)
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _sample_lam(self) -> float:
        if self.alpha > 0:
            lam = float(np.random.beta(self.alpha, self.alpha))
        else:
            lam = 1.0
        if self.force_dominant:
            lam = max(lam, 1.0 - lam)
        return lam

    def _cutmix_temporal(self, x: torch.Tensor, x_b: torch.Tensor, lam: float):
        """CutMix sobre el eje tiempo (último eje) ajustando lam.

        x, x_b: (B, C, F, T). Devuelve (mixed_x, lam_actual) donde
        ``lam_actual = 1 - cut_len / T`` representa la fracción de x preservada.
        """
        T = x.size(-1)
        cut_len = max(1, int(round(T * (1.0 - lam))))
        cut_len = min(cut_len, T)
        cut_start = int(np.random.randint(0, T - cut_len + 1))

        mixed_x = x.clone()
        mixed_x[..., cut_start:cut_start + cut_len] = x_b[..., cut_start:cut_start + cut_len]
        lam_actual = 1.0 - (cut_len / T)
        return mixed_x, lam_actual

    def forward(self, x, y, x_pseudo=None, y_pseudo=None, is_pseudo_mix: bool = False):
        if self.mixup_prob <= 0:
            return x, y
        if self._epoch < self.warmup_epochs:
            return x, y
        if np.random.rand() > self.mixup_prob:
            return x, y

        batch_size = x.size(0)
        y = y.float()

        if is_pseudo_mix and x_pseudo is not None and y_pseudo is not None:
            if x_pseudo.size(0) < batch_size:
                return x, y
            x_b = x_pseudo[:batch_size]
            y_b = y_pseudo[:batch_size].float()
        else:
            index = torch.randperm(batch_size, device=x.device)
            x_b = x[index]
            y_b = y[index]

        lam = self._sample_lam()
        use_cutmix = (self.cutmix_prob > 0.0) and (np.random.rand() < self.cutmix_prob)

        if use_cutmix:
            mixed_x, lam_actual = self._cutmix_temporal(x, x_b, lam)
        else:
            mixed_x = lam * x + (1.0 - lam) * x_b
            lam_actual = lam

        mixed_y = lam_actual * y + (1.0 - lam_actual) * y_b
        mixed_y = mixed_y.clamp(0.0, 1.0)

        return mixed_x, mixed_y
