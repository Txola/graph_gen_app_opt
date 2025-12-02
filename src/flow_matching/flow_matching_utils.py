import torch
from torch.nn import functional as F
from utils import PlaceHolder


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, (
        "Variables not masked properly."
    )


def sample_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_X = (
        x_limit.flatten(end_dim=-2).multinomial(1, replacement=True).reshape(bs, n_max)
    )
    U_E = (
        e_limit.flatten(end_dim=-2)
        .multinomial(1, replacement=True)
        .reshape(bs, n_max, n_max)
    )
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)


def sample_discrete_features(probX, probE, node_mask, mask=False):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    """
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1, replacement=True)  # (bs * n, 1)
    # X_t = Categorical(probs=probX).sample()  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1, replacement=True).reshape(bs, n, n)  # (bs, n, n)
    # E_t = Categorical(probs=probE).sample().reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    if mask:
        X_t = X_t * node_mask
        E_t = E_t * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


def p_xt_g_x1(X1, E1, t, limit_dist):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value
    device = X1.device
    limit_dist.X = limit_dist.X.to(device)
    limit_dist.E = limit_dist.E.to(device)

    t_time = t.squeeze(-1)[:, None, None]
    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

    Xt = t_time * X1_onehot + (1 - t_time) * limit_dist.X[None, None, :]
    Et = (
        t_time[:, None] * E1_onehot
        + (1 - t_time[:, None]) * limit_dist.E[None, None, None, :]
    )

    assert ((Xt.sum(-1) - 1).abs() < 1e-4).all() and (
        (Et.sum(-1) - 1).abs() < 1e-4
    ).all()

    return Xt.clamp(min=0.0, max=1.0), Et.clamp(min=0.0, max=1.0)


def dt_p_xt_g_x1(X1, E1, limit_dist):
    # x1 (B, D)
    # returns (B, D, S) for varying x_t value
    device = X1.device
    limit_dist.X = limit_dist.X.to(device)
    limit_dist.E = limit_dist.E.to(device)

    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

    dX = X1_onehot - limit_dist.X[None, None, :]
    dE = E1_onehot - limit_dist.E[None, None, None, :]

    assert (dX.sum(-1).abs() < 1e-4).all() and (dE.sum(-1).abs() < 1e-4).all()

    return dX, dE
