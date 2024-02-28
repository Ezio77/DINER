def relative_l2_loss(output,targets):
    output = (output + 1) / 2.
    targets = (targets + 1) / 2.
    relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
    loss = relative_l2_error.mean()
    return loss

