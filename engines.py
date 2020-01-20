import torch
from ignite.engine import _prepare_batch, Engine


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch,
                              output_transform=lambda x, y, y_pred, loss: loss.item(), accumulation_steps=8):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)

        loss = loss_fn(y_pred, y) / accumulation_steps
        loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return output_transform(x, y, y_pred, loss)

    return Engine(_update)