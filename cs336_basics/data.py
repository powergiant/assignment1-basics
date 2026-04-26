import numpy.typing as npt
import torch
import numpy

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    size = dataset.size
    ran = size - context_length
    starts = numpy.random.randint(0, ran, size=batch_size).tolist()

    samples = numpy.array([dataset[start:start+context_length+1] for start in starts])

    samples = torch.tensor(samples, device=device)

    input = samples[:, :-1]
    target = samples[:, 1:]

    return input, target


