"""
Changes from torch.nn.utils.convert_parameters

- Patched vector_to_parameters to use: `param.data.copy_` instead of `param.data =` to
preserve the memory_format of param.data

- Added parameters_to_vector_device accept a device parameter.
    -- Note that `memory_format` must be contiguous for `.view(-1)`, otherwise you will get an error.

- Added foreach-based helpers (parameters_to_buffer_row_, buffer_row_to_parameters_)
  that avoid per-step allocation by writing directly into a pre-allocated row of a
  ring buffer using ``torch._foreach_copy_`` (single fused kernel launch).
"""

import torch


def parameters_to_vector_device(parameters, device):
    vec = []
    for param in parameters:
        vec.append(param.detach().to(device=device, memory_format=torch.contiguous_format).view(-1))
    return torch.cat(vec)


def _slice_views(row, parameters):
    """Build a list of strided views of ``row`` matching each param's shape."""
    targets = []
    offset = 0
    for p in parameters:
        n = p.numel()
        targets.append(row[offset:offset + n].view_as(p))
        offset += n
    return targets


def parameters_to_buffer_row_(parameters, row):
    """Copy ``parameters`` into the contiguous 1-D ``row`` in a single fused kernel.

    ``row`` must be a contiguous 1-D tensor with ``row.numel() == sum(p.numel())``.
    Cross-device / cross-dtype writes are handled by ``copy_`` semantics.
    """
    parameters = list(parameters)
    targets = _slice_views(row, parameters)
    sources = [p.detach() for p in parameters]
    torch._foreach_copy_(targets, sources)


def buffer_row_to_parameters_(row, parameters):
    """Copy a contiguous 1-D ``row`` back into ``parameters`` (single fused kernel)."""
    parameters = list(parameters)
    sources = _slice_views(row, parameters)
    targets = [p.data for p in parameters]
    torch._foreach_copy_(targets, sources)


def vector_to_parameters(vec, parameters):
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        # *** This line changed from `param.data =` to `param.data.copy_`
        param.data.copy_(vec[pointer:pointer + num_param].view_as(param).data)

        # Increment the pointer
        pointer += num_param
