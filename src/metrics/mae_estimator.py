import pandas as pd


class MAEEstimator:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        df["mae_no_exit"] = df["mae_no_exit"].apply(
            lambda t: float(str(t).replace("tensor(", "").replace(")", ""))
        )
        grouped = df.groupby("steps")["mae_no_exit"].mean().reset_index()

        self.steps = grouped["steps"].tolist()
        self.maes = grouped["mae_no_exit"].tolist()

    def estimate(self, step):
        """Estimate MAE at a given step using linear interpolation."""
        if step in self.steps:
            return self.maes[self.steps.index(step)]

        if step < self.steps[0] or step > self.steps[-1]:
            raise ValueError("Step out of bounds for MAE estimation.")

        for i in range(len(self.steps) - 1):
            x0, x1 = self.steps[i], self.steps[i + 1]
            if x0 <= step <= x1:
                y0, y1 = self.maes[i], self.maes[i + 1]
                return y0 + (step - x0) / (x1 - x0) * (y1 - y0)
