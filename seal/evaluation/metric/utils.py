import torch

def get_constr_out(x, R) -> torch.Tensor:
    r"""
        Given the output of the neural network x, this function 
        returns the output of MCM given the hierarchy constraint 
        expressed in the adjacency matrix R. 
    """

    c_out = x
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, max_index = torch.max(R_batch * c_out.double(), dim = 2)
    return final_out
