import os

import pandas as pd
from models.hyperparameter_model_MLP import VUNPredictorMLP


class LookupTable:
    """
    Loads an existing lookup table or generates a new one if missing.
    Allows querying best hyperparameters for a given number of diffusion steps.
    """

    def __init__(
        self,
        lookup_csv="../csvs/lookup_table.csv",
        training_csv="../csvs/config_results_fixed.csv",
        num_steps_list=None,
        generator_kwargs=None,
    ):
        """
        Initializes the LookupTable by loading or generating the lookup table.
        """

        if generator_kwargs is None:
            generator_kwargs = {}

        if num_steps_list is None:
            num_steps_list = list(range(25, 70))

        if os.path.exists(lookup_csv):
            print("Loading existing lookup table...")
            self.table = pd.read_csv(lookup_csv)
            print("Lookup table loaded.")
        else:
            print("Lookup table does not exist. Generating new one...")

            generator = LookupTableGenerator(
                csv_path=training_csv, **generator_kwargs
            )
            self.table = generator.generate_predictions(
                num_steps_list=num_steps_list,
            )

    def get_best_params(self, num_steps):
        """
        Returns the best hyperparameters for the number of steps.
        """

        if num_steps in self.table["num_steps"].values:
            return self.table[self.table["num_steps"] == num_steps].iloc[0].to_dict()
        raise ValueError(f"num_steps {num_steps} not found in lookup table.")


class LookupTableGenerator:
    def __init__(
        self,
        csv_path,
        hidden_sizes=[32, 32],
        lr=1e-3,
        epochs=500,
        batch_size=32,
        device=None,
    ):
        """
        Loads gridsearch CSV, initializes the MLP predictor.
        """
        print(f"Loading training data from: {csv_path}")
        self.df = pd.read_csv(csv_path)

        # Initialize the MLP model
        self.model = VUNPredictorMLP(
            df_existing=self.df,
            hidden_sizes=hidden_sizes,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )

    def generate_predictions(
        self, num_steps_list, out_csv="predicted_lookup_table_mlp.csv"
    ):
        """
        Generates a CSV where for each num_steps in num_steps_list,
        the best eta, omega, distortion combination is computed using optimize_params.
        Only creates the CSV if it does not already exist.
        """

        # if the file already exists, just load and return it
        if os.path.exists(out_csv):
            print(
                f"[INFO] The file '{out_csv}' already exists. Results are already stored."
            )
            print("[INFO] Skipping MLP prediction generation.")
            return pd.read_csv(out_csv)

        # if the file does not exist, generate it
        results = []

        trial_prediction = self.model.predict_validity(
            num_steps=10, distortion="polydec", eta=0, omega=0.5
        )

        print(
            f"[INFO] Sample prediction for num_steps=10, distortion='polydec', eta=0, omega=0.5: validity = {trial_prediction:.4f}"
        )

        print(f"[INFO] Generating new predictions and saving to: {out_csv}")

        print("Generating predictions with MLP...")
        for nsteps in num_steps_list:
            print(f"  -> Optimizing for num_steps = {nsteps}")
            best_cfg = self.model.optimize_params(num_steps=nsteps)

            if best_cfg is not None:
                results.append(best_cfg)
            else:
                print(f"[WARNING] No optimal result found for num_steps = {nsteps}")

        df_results = pd.DataFrame(results)
        df_results.to_csv(out_csv, index=False)

        print(f"[INFO] Predictions saved to: {out_csv}")
        return df_results
