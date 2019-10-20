import torch


def hess_largest_eigenvalue(criterion, inputs, target, net, device, tolerance=1e-6, max_iters=100):
    # Create gradients vector

    # Create v
    dim = 0
    for w in net.parameters():
        dim += w.numel()
    v = torch.normal(torch.zeros(dim), 1).to(device)

    iter_num = 0
    lmbda = 9999
    lmbda_prev = 999
    torch.autograd.set_grad_enabled(True)
    print('Iterating')
    while abs(lmbda_prev - lmbda) > tolerance and iter_num < max_iters:
        print('.', end='')
        iter_num += 1
        # Normalize v
        v.div_(v.norm(p=2))

        # -- Calculate gradients
        # Forward
        outputs = net(inputs)
        # Max trick:
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            outputs = outputs - torch.max(outputs, dim=1)[0].view(torch.max(outputs, dim=1)[0].shape[0], 1
                                                                  ).repeat(1, outputs.shape[1])
        loss = criterion(outputs, target)

        # Backward
        net.zero_grad()
        loss.backward(create_graph=True)

        # Create gradients vector
        g_vector = None
        grads = [w.grad for w in net.parameters()]
        for g in grads:
            g_vector = g.contiguous().view(-1) if g_vector is None else torch.cat([g_vector, g.view(-1)])

        # -- Calculate Hv (H is the Hessian)
        hv = torch.dot(v, g_vector)
        hv = torch.autograd.grad(hv, net.parameters())
        hv_vector = None
        for h in hv:
            hv_vector = h.contiguous().view(-1) if hv_vector is None else torch.cat([hv_vector,
                                                                                     h.contiguous().view(-1)])

        # -- Set lambda
        lmbda_prev = lmbda
        lmbda = torch.dot(hv_vector, v)
        v = hv_vector.detach()
    torch.autograd.set_grad_enabled(False)
    print("Stopped on iter " + str(iter_num) + ", lmbda=" + str(lmbda.item()))
    return lmbda.item()
