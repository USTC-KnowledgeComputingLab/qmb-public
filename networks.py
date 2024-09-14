import argparse
import logging
import naqs as naqs_m
import attention as attention_m


def naqs(model, input_args):
    logging.info("parsing args %a by network naqs", input_args)
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--hidden-width", dest="hidden", type=int, default=[512], nargs="+", help="hidden width of the network")
    parser.add_argument("--help-network", action="help")
    args = parser.parse_args(input_args)

    hidden = args.hidden
    logging.info("hidden is set to %a", hidden)

    network = naqs_m.WaveFunction(
        double_sites=model.n_qubits,
        physical_dim=2,
        is_complex=True,
        spin_up=model.n_electrons // 2,
        spin_down=model.n_electrons // 2,
        hidden_size=hidden,
        ordering=+1,
    ).double()

    return network


def attention(model, input_args):
    logging.info("parsing args %a by network transformers", input_args)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding-dim", dest="embedding_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("-m", "--heads-num", dest="heads_num", type=int, default=8, help="heads number")
    parser.add_argument("-f", "--feed-forward-dim", dest="feed_forward_dim", type=int, default=2048, help="feedforward dimension")
    parser.add_argument("-d", "--depth", dest="depth", type=int, default=6, help="network depth")
    parser.add_argument("--help-network", action="help")
    args = parser.parse_args(input_args)

    embedding_dim = args.embedding_dim
    heads_num = args.heads_num
    feed_forward_dim = args.feed_forward_dim
    depth = args.depth
    logging.info("embedding dim: %d, heads_num: %d, feed forward dim: %d, depth: %d", embedding_dim, heads_num, feed_forward_dim, depth)

    network = attention_m.WaveFunction(
        double_sites=model.n_qubits,
        physical_dim=2,
        is_complex=True,
        spin_up=model.n_electrons // 2,
        spin_down=model.n_electrons // 2,
        embedding_dim=embedding_dim,
        heads_num=heads_num,
        feed_forward_dim=feed_forward_dim,
        depth=depth,
        ordering=+1,
    ).double()

    return network
