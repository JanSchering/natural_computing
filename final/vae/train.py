import tqdm
from model import Model

def train(model: Model, n_updates=int(1e6), eval_interval=1000, suffix:str="VAE") -> None:
    best = float("inf")
    for i in tqdm.tqdm(range(n_updates)):
        model.train_batch()
        if (i + 1) % eval_interval == 0:
            loss = model.eval_batch()
            model.save(f"latest_{suffix}")
            if loss < best:
                best = loss
                model.save(f"best_{suffix}")