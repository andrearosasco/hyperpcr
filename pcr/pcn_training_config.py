import torch

class Config:

    class General:
        device = 'cuda'
        visible_dev = '0'
        seed = 1
        num_workers = 30

    class Train:
        lr = 1e-5
        wd = 0.0
        n_epoch = 1000
        clip_value = 1
        optimizer = torch.optim.Adam
        loss = torch.nn.BCEWithLogitsLoss
        loss_reduction = "mean"  # "none"

        adaptation = False
        chamfer = False

        mb_size = 32

    class Eval:
        wandb = True
        log_metrics_every = 100
        val_every = 1

        grid_eval = False
        grid_res_step = 0.04

        mb_size = 32

    class Data:
        dataset_path = 'data/MCD'
        partial_points = 2048  # number of points per input
        multiplier_complete_sampling = 50
        implicit_input_dimension = 8192 * 2
        dist = [0.1, 0.4, 0.5]
        noise_rate = 0.1
        tolerance = 0.005
        n_classes = 55
        # bring the partial pcd closer to max-z (i.e. 0.5) so that the reconstruction points are in a 0.5 cube centered
        # in 0, 0, 0

        class Train:
            mb_size = 32

        class Eval:
            mb_size = 32

    class Model:
        knn_layer = 1
        # Transformer
        n_channels = 3
        embed_dim = 384
        encoder_depth = 6
        mlp_ratio = 2.
        qkv_bias = False
        num_heads = 6
        attn_drop_rate = 0.
        drop_rate = 0.
        qk_scale = None
        out_size = 1024
        # Implicit Function
        hidden_dim = 32
        depth = 2
        # Others
        use_object_id = False
        use_deep_weights_generator = False
        assert divmod(embed_dim, num_heads)[1] == 0

