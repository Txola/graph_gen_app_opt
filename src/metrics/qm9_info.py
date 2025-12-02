import torch
import utils as utils


class DistributionNodes:
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) is dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p


class QM9Infos:
    def __init__(self):
        self.name = "qm9"
        self.atom_encoder = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        self.atom_decoder = ["H", "C", "N", "O", "F"]
        self.max_n_nodes = 29
        self.n_nodes = torch.tensor(
            [
                0,
                0,
                0,
                1.5287e-05,
                3.0574e-05,
                3.8217e-05,
                9.1721e-05,
                1.5287e-04,
                4.9682e-04,
                1.3147e-03,
                3.6918e-03,
                8.0486e-03,
                1.6732e-02,
                3.0780e-02,
                5.1654e-02,
                7.8085e-02,
                1.0566e-01,
                1.2970e-01,
                1.3332e-01,
                1.3870e-01,
                9.4802e-02,
                1.0063e-01,
                3.3845e-02,
                4.8628e-02,
                5.4421e-03,
                1.4698e-02,
                4.5096e-04,
                2.7211e-03,
                0.0000e00,
                2.6752e-04,
            ]
        )
        self.valencies = [1, 4, 3, 2, 1]
        self.max_weight = 390
        self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}

        self.node_types = torch.tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
        self.edge_types = torch.tensor([0.88162, 0.11062, 5.9875e-03, 1.7758e-03, 0])

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = {"X": 19, "E": 17, "y": 8}
        self.output_dims = {"X": 5, "E": 5, "y": 0}
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
        )

        example_data = {
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": example_batch["y"],
            "node_mask": node_mask,
        }
        self.input_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": example_batch["y"].size(1)
            + 1,  # this part take into account the conditioning
        }  # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims["X"] += ex_extra_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims["X"] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": 0,
        }
