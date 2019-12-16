from torch import optim


class Adam(optim.Adam):
    @classmethod
    def resolve_args(cls, args, params):
        options = {}
        options['lr'] = args.get("learning_rate", 0.01)
        options['weight_decay'] = args.get("weight_decay", 0)
        # options['lr_decay'] = args.get("lr_decay", 0)
        options['betas'] = args.get("betas", (0.9, 0.999))
        options['eps'] = args.get("eps_", 1e-08)

        return cls(params, **options)
