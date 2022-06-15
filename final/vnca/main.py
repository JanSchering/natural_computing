from train import train
from init import init_vnca

if __name__ == "__main__":
    vnca = init_vnca()
    vnca.eval_batch()
    train(vnca, n_updates=100_000, eval_interval=100)
    vnca.test(128)
