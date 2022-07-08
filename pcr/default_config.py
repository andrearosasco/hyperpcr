import torch

class Config:

    class General:
        device = 'cuda'
        visible_dev = '0'
        seed = 1

    class Processing:
        use_offset = True
        scale = 0.7

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

        class Decoder:
            num_points = 8192 * 2
            thr = 0.7
            itr = 20

