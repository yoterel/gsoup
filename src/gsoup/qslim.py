import numpy as np


def remove_unreferenced_vertices(v, f):
    referenced = np.zeros((v.shape[0]), dtype=np.bool)
    referenced[f.flatten()] = True
    idx = -1 * np.ones((v.shape[0]), dtype=np.int32)
    idx[referenced] = np.arange(np.sum(referenced), dtype=np.int32)
    f = idx[f.flatten()].reshape((-1, f.shape[1]))
    v = v[referenced]
    return v, f, idx


def get_edges(f):
    e1 = np.concatenate((f[:, 0], f[:, 1]), axis=-1)
    e2 = np.concatenate((f[:, 1], f[:, 2]), axis=-1)
    e3 = np.concatenate((f[:, 2], f[:, 3]), axis=-1)
    e = np.concatenate((e1, e2, e3), axis=-1)
    e = np.sort(e, axis=-1)
    e = np.unique(e, axis=0)
    return e


def edge_contraction(v, f, v1, v2, v_hat):
    # replace all instances of v1 with v2 in f
    f[f==v1] = v2
    # remove degenerate triangles
    mask = np.ones(f.shape[0], dtype=np.bool)
    f = f[mask] 
    # place v2 in v_hat
    v[v2] = v_hat
    # remove v1 from v
    v, f, _ = remove_unreferenced_vertices(v, f)
    return v, f


def compute_quadtratic_cost(v, f):
    e1 = np.concatenate((f[:, 0], f[:, 1]), axis=-1)
    e2 = np.concatenate((f[:, 1], f[:, 2]), axis=-1)
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n, axis=-1)
    d = np.array([0, 0, 0, -v[f[:, 0]].dot(n)])
    Q = np.eye((v.shape[0], 4, 4))
    Q[:, :3, :3] = n[:, None, :] @ n[:, :, None]
    Q[:, -1, :] = d
    Q[:, :, -1] = d
    return Q


def compute_costs(v, Qs, valid_pairs):
    return np.sum(v.T[valid_pairs] @ Qs[valid_pairs] @ v[valid_pairs])


def compute_v_hat(v, f, edge_to_contract, Qs):
    Q_hat = np.sum(Qs[edge_to_contract])
    if np.det(Q_hat) != 0:
        return np.linalg.inv(Q_hat) @ np.array([0, 0, 0, 1])
    else:
        return (v[edge_to_contract[0]] + v[edge_to_contract[1]]) / 2


def qslim(v, f, budget):
    if budget < 3:
        raise ValueError("minimum budget is 3 triangles")
    while f.shape[0] > budget:
        valid_pairs = get_edges(f)
        Qs = compute_quadtratic_cost(v, f)
        costs = compute_costs(v, Qs, valid_pairs)
        lowest_cost = np.argmin(costs)
        edge_to_contract = valid_pairs[lowest_cost]
        v_hat = compute_v_hat(v, f, edge_to_contract, Qs)
        v, f = edge_contraction(v, f, edge_to_contract, v_hat)

