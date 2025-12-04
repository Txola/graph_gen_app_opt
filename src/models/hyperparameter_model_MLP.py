import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class VUNPredictorMLP:
    def __init__(
        self,
        df_existing,
        hidden_sizes=[32, 32],
        lr=1e-3,
        epochs=500,
        batch_size=32,
        device=None,
    ):
        """
        Initialices and trains an MLP to predict vun based on existing data.
        """
        self.df = df_existing.copy()
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # One-hot encode de distorsión
        self.enc = OneHotEncoder(sparse_output=False)
        dist_encoded = self.enc.fit_transform(self.df[["distortion"]])

        # Construir X y y
        X = np.column_stack(
            [
                self.df["num_steps"].values,
                dist_encoded,
                self.df[["eta", "omega"]].values,
            ]
        )
        print(f"Input feature shape: {X.shape}")
        print(f"X sample:\n{X[:5]}")
        y = self.df["validity_mean"].values.reshape(-1, 1)

        print(f"y sample:\n{y[:5]}")
        print(f"y shape: {y.shape}")

        # Escalado
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        Xn = self.X_scaler.fit_transform(X)
        yn = self.y_scaler.fit_transform(y)

        self.Xn = torch.tensor(Xn, dtype=torch.float32).to(self.device)
        self.yn = torch.tensor(yn, dtype=torch.float32).to(self.device)

        input_size = Xn.shape[1]
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size

        print(f"Training MLP at {self.device}...")
        self._train()
        print("MLP training complete.")

    def _train(self):
        dataset = torch.utils.data.TensorDataset(self.Xn, self.yn)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                y_pred = self.model(xb)
                loss = self.criterion(y_pred, yb)
                loss.backward()
                self.optimizer.step()

    def predict_validity(self, num_steps, distortion, eta, omega):
        """
        Predicts validity for given parameters using the trained MLP.
        """
        dist_vec = np.zeros(len(self.enc.categories_[0]))
        dist_idx = list(self.enc.categories_[0]).index(distortion)
        dist_vec[dist_idx] = 1.0

        x_raw = np.concatenate([[num_steps], dist_vec, [eta, omega]]).reshape(1, -1)
        x_scaled = self.X_scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred_norm = self.model(x_tensor).cpu().numpy()[0, 0]
        y_pred = self.y_scaler.inverse_transform([[y_pred_norm]])[0, 0]
        return y_pred

    def optimize_params(self, num_steps):
        """
        Finds the optimal eta and omega to maximize validity for a given number of steps.
        """

        eta_bounds = (self.df["eta"].min(), self.df["eta"].max())
        omega_bounds = (self.df["omega"].min(), self.df["omega"].max())
        bounds = [eta_bounds, omega_bounds]

        best_val = -np.inf
        best_cfg = None

        for distortion in self.enc.categories_[0]:

            def obj_fn_global(x):
                eta, omega = x
                return -self.predict_validity(
                    num_steps=num_steps, distortion=distortion, eta=eta, omega=omega
                )

            res = differential_evolution(obj_fn_global, bounds)

            print(
                f"  Distortion: {distortion}, Best params: eta={res.x[0]:.4f}, omega={res.x[1]:.4f}, Predicted validity={-res.fun:.4f}"
            )

            if res.success:
                mu_pred = -res.fun
                if mu_pred > best_val:
                    best_val = mu_pred
                    best_cfg = {
                        "num_steps": num_steps,
                        "distortion": distortion,
                        "eta": res.x[0],
                        "omega": res.x[1],
                        "pred_score": mu_pred,
                    }

        return best_cfg
