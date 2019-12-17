config = {
    'pretrained_ep': 0, #not_implemented
    'continue_training': False, # not implemented

    #model_settings
    'multichoice': True,
    'model_name': 'dbertqa', # 'twopunch', dbertqa
    'aggregation': 'cat', # aggregate image and text with this operation in vaswani.py, twopunch.py
        # sum
        # elemwise
        # hadamard: not implemented


    'extractor_batch_size': 384,
    'log_path': 'data/log',
    'batch_sizes': (12, 12, 12),
    'lower': True,
    'use_inputs': [],  # We advise not to use description for the challenge
    'cache_image_vectors': True,
    'image_path': 'data/AnotherMissOh/AnotherMissOh_images',
    'data_path': 'data/AnotherMissOh/AnotherMissOh_QA/AnotherMissOhQA_set_subtitle.jsonl',
    'subtitle_path': 'data/AnotherMissOh/AnotherMissOh_subtitles.json',

    'tokenizer': 'distilbert', #'nltk',
    'vocab_freq': 1,
    #'vocab_pretrained': "glove.6B.300d",
    'video_type': ['shot', 'scene'],
    'feature_pooling_method': 'mean',
    'allow_empty_images': False,
    'num_workers': 40,

    'image_dim': 512,  # hardcoded for ResNet50
    'nsample': 8,
    'n_dim': 512,


    'layers': 3,
    'dropout': 0.5,
    'learning_rate': 5e-4,
    'loss_name': 'cross_entropy_loss',
    'optimizer': 'adam',
    # 'metrics': ['bleu', 'rouge'],
    'metrics': [],
    'log_cmd': False,
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'text_feature_names': ['subtitle', 'description'],



    ## vaswani model params
    'rel_positional_emb': 4, #we just use ~8 images for image encoder
    'N': 6, #number of transformer block repetition
    'd_ff': 512, #dimension of feedforward for tr blocks
    'h': 8, #number of heads for multihead attention
    'dropout_tr': 0.1, # dropout_ratio for transformer

    #replaced with image_dim
    #'d_model': 512,



    #lrschedulers.
    'lrschedule': 'rop', # None, 'rop', 'cos', 'sharp_cos', 'lin'
    'warmup_steps': 4000, # common option for cos and lin schedule
    'max_epochs': 20,
    ## rop options
    'gamma': 0.5,
    'patience': 2,
    'mode': 'min',
    'threshold':1e-3,
    'threshold_mode': 'rel',
    'min_lr':1e-8,
    'eps':1e-8, # epsilon for rop

    #'num_training_steps': 200000, # calculated from iterators and batches

}


debug_options = {
    # 'image_path': './data/images/samples',
}

log_keys = [
    'model_name',
    'feature_pooling_method',
]
