import torch


class TimeDistorter:
    def __init__(
        self,
        sample_distortion,
    ):
        self.sample_distortion = sample_distortion  # used for get_ft

    def sample_ft(self, t):
        t_distort = self.apply_distortion(t, self.sample_distortion)
        return t_distort

    def apply_distortion(self, t, distortion_type):
        assert torch.all((t >= 0) & (t <= 1)), "t must be in the range (0, 1)"

        if distortion_type == "identity":
            ft = t
        elif distortion_type == "cos":
            ft = (1 - torch.cos(t * torch.pi)) / 2
        elif distortion_type == "revcos":
            ft = 2 * t - (1 - torch.cos(t * torch.pi)) / 2
        elif distortion_type == "polyinc":
            ft = t**2
        elif distortion_type == "polydec":
            ft = 2 * t - t**2
        elif distortion_type.startswith("new_polydec_"):
            a = float(distortion_type.split("_")[-1])
            ft = 1 - (1 - t) ** a
        elif distortion_type.startswith("new_cos_"):
            a = float(distortion_type.split("_")[-1])
            ft = t**a / (t**a + (1 - t) ** a)
        elif distortion_type.startswith("log_"):
            a = torch.tensor(float(distortion_type.split("_")[-1]))
            t = torch.tensor(t)
            ft = torch.log(1 + a * t) / torch.log(1 + a)
        elif distortion_type == "beta":
            raise ValueError(f"Unsupported for now: {distortion_type}")
        elif distortion_type == "logitnormal":
            raise ValueError(f"Unsupported for now: {distortion_type}")
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

        return ft
