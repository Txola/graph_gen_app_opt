import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from flow_matching import flow_matching_utils
from flow_matching.noise_distribution import NoiseDistribution
from flow_matching.rate_matrix import RateMatrixDesigner
from flow_matching.time_distorter import TimeDistorter
from models.transformer_model import GraphTransformer


def load_transformer_model(cfg, qm9_dataset_infos, device):
    model = GraphTransformer(
        n_layers=cfg.model.n_layers,
        input_dims=qm9_dataset_infos.input_dims,
        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        hidden_dims=cfg.model.hidden_dims,
        output_dims=qm9_dataset_infos.output_dims,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
    )

    state = torch.load(cfg.model.checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


class QM9CondSampler:
    def __init__(
        self,
        cfg,
        qm9_dataset_infos,
        extra_features,
        domain_features,
        model,
        evaluator,
        eta,
        omega,
        distortion,
    ):
        self.cfg = cfg
        self.omega = omega
        self.device = next(model.parameters()).device
        self.node_dist = qm9_dataset_infos.nodes_dist

        self.extra_features = extra_features
        self.domain_features = domain_features

        self.noise_dist = NoiseDistribution(cfg.model.transition, qm9_dataset_infos)
        self.limit_dist = self.noise_dist.get_limit_dist()

        self.noise_dist.update_input_output_dims(qm9_dataset_infos.input_dims)
        self.noise_dist.update_dataset_infos(qm9_dataset_infos)

        self.model = model
        self.evaluator = evaluator

        self.time_distorter = TimeDistorter(
            sample_distortion=distortion,
        )

        self.rate_matrix_designer = RateMatrixDesigner(
            rdb=self.cfg.sample.rdb,
            rdb_crit=self.cfg.sample.rdb_crit,
            eta=eta,
            omega=omega,
            limit_dist=self.limit_dist,
        )

        self.initialized = False
        self.finished = False

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def initialize(
        self,
        batch_size,
        sample_steps,
        condition_value,
        early_exit=False,
        num_nodes=None,
    ):
        """
        Prepares internal state so the sampler can run step-by-step.
        After this step() can be called repeatedly until finished=True.

        condition_value (float or tuple of two floats): If a single float,
        sets the same conditional value for all batch elements. If a tuple,
        generates a linspace between the two limits for each batch element.
        """
        self.initialized = True
        self.finished = False
        self.t_int = 0

        self.batch_size = batch_size
        self.sample_steps = sample_steps
        self.conditional = condition_value is not None
        self.condition_value = condition_value
        self.early_exit = early_exit

        if num_nodes is None:
            self.n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif isinstance(num_nodes, int):
            self.n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            self.n_nodes = num_nodes
        self.n_max = torch.max(self.n_nodes).item()

        arange = (
            torch.arange(self.n_max, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        self.node_mask = arange < self.n_nodes.unsqueeze(1)

        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=self.node_mask
        )
        if self.conditional:
            if isinstance(condition_value, (tuple, list)) and len(condition_value) == 2:
                condition = torch.linspace(
                    condition_value[0],
                    condition_value[1],
                    steps=batch_size,
                    device=self.device,
                ).unsqueeze(1)
            else:
                condition = torch.tensor(
                    [condition_value], device=self.device
                ).unsqueeze(0)
                condition = condition.repeat(batch_size, 1)
            z_T.y = condition

        self.X, self.E, self.y = z_T.X, z_T.E, z_T.y

        self.last_discrete = None

    @torch.no_grad()
    def step(self):
        """
        Performs exactly one sampling step.
        This allows round-robin scheduling across many jobs.
        """
        if not self.initialized or self.finished:
            return

        batch_size = self.batch_size
        t_int = self.t_int
        sample_steps = self.sample_steps
        node_mask = self.node_mask
        n_nodes = self.n_nodes

        t_array = t_int * torch.ones((batch_size, 1)).type_as(self.y)
        t_norm = t_array / (sample_steps)
        if ("absorb" in self.cfg.model.transition) and (t_int == 0):
            t_norm = t_norm + 1e-6
        s_array = t_array + 1
        s_norm = s_array / (sample_steps)

        t_norm = self.time_distorter.sample_ft(t_norm)
        s_norm = self.time_distorter.sample_ft(s_norm)

        sampled_s, sampled_discrete = self.sample_p_zs_given_zt(
            t_norm, s_norm, self.X, self.E, self.y, node_mask
        )

        if self.early_exit:
            self.X, self.E, self.y = (
                sampled_discrete.X,
                sampled_discrete.E,
                sampled_discrete.y,
            )
            n = n_nodes[0]
            atom_types = self.X[0, :n].cpu()
            edge_types = self.E[0, :n, :n].cpu()
            valid = self.evaluator.compute_validity([[atom_types, edge_types]])
            if valid == 1.0:
                self.finished = True
                self.last_discrete = sampled_discrete
                return

        self.X, self.E, self.y = sampled_s.X, sampled_s.E, sampled_s.y
        self.last_discrete = sampled_discrete

        self.t_int += 1
        if self.t_int >= sample_steps:
            self.finished = True

    @torch.no_grad()
    def finalize(self):
        """Returns the generated graph after all steps are completed."""
        if self.last_discrete is None:
            raise RuntimeError("No sampling performed")

        X, E, y = self.last_discrete.X, self.last_discrete.E, self.last_discrete.y
        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)

        samples = []
        labels = []
        for i in range(self.batch_size):
            n = self.n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            samples.append([atom_types, edge_types])
            labels.append(y[i].cpu())

        return samples, labels

    def sample(
        self,
        batch_size,
        sample_steps,
        condition_value,
        early_exit=False,
        num_nodes=None,
    ):
        """Full sampling function that runs all steps internally."""
        self.initialize(
            batch_size, sample_steps, condition_value, early_exit, num_nodes
        )

        for _ in range(sample_steps):
            if self.finished:
                break
            self.step()

        return self.finalize()

    def compute_step_probs(self, R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e):
        step_probs_X = R_t_X * dt  # type: ignore # (B, D, S)
        step_probs_E = R_t_E * dt  # (B, D, S)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        # assert (E_t.argmax(-1) < 4).all()
        step_probs_X.scatter_(-1, X_t.argmax(-1)[:, :, None], 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1)[:, :, :, None], 0.0)

        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs_X.scatter_(
            -1,
            X_t.argmax(-1)[:, :, None],
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            E_t.argmax(-1)[:, :, :, None],
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        # step 2 - merge to the original formulation
        prob_X = step_probs_X.clone()
        prob_E = step_probs_E.clone()

        return prob_X, prob_E

    def sample_p_zs_given_zt(
        self,
        t,
        s,
        X_t,
        E_t,
        y_t,
        node_mask,
        # , condition
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, dx = X_t.shape
        _, _, _, de = E_t.shape
        dt = (s - t)[0]

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        limit_x = self.limit_dist.X
        limit_e = self.limit_dist.E

        G_1_pred = pred_X, pred_E
        G_t = X_t, E_t

        R_t_X, R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(
            t,
            node_mask,
            G_t,
            G_1_pred,
        )

        if self.conditional:
            uncond_y = torch.ones_like(y_t, device=self.device) * -1
            noisy_data["y_t"] = uncond_y

            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)

            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

            G_1_pred = pred_X, pred_E

            R_t_X_uncond, R_t_E_uncond = (
                self.rate_matrix_designer.compute_graph_rate_matrix(
                    t,
                    node_mask,
                    G_t,
                    G_1_pred,
                )
            )

            guidance_weight = self.omega
            R_t_X = torch.exp(
                torch.log(R_t_X_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_X + 1e-6) * guidance_weight
            )
            R_t_E = torch.exp(
                torch.log(R_t_E_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_E + 1e-6) * guidance_weight
            )

        prob_X, prob_E = self.compute_step_probs(
            R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e
        )

        if s[0] == 1.0:
            prob_X, prob_E = pred_X, pred_E

        sampled_s = flow_matching_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_x)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        out_discrete = out_discrete.mask(node_mask, collapse=True).type_as(y_t)

        return out_one_hot, out_discrete

    def compute_extra_data(self, noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        extra_features = self.extra_features(noisy_data)

        # one additional category is added for the absorbing transition
        X, E, y = self.noise_dist.ignore_virtual_classes(
            noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        )
        noisy_data_to_mol_feat = noisy_data.copy()
        noisy_data_to_mol_feat["X_t"] = X
        noisy_data_to_mol_feat["E_t"] = E
        noisy_data_to_mol_feat["y_t"] = y
        extra_molecular_features = self.domain_features(noisy_data_to_mol_feat)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
