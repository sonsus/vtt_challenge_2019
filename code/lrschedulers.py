'''
from pytorch-huggingface
https://huggingface.co/transformers/_modules/transformers/optimization.html#get_constant_schedule_with_warmup
Apache-2.0 license need to be clarified
'''

#lrschedulers.py
import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import math

def get_scheduler(args, optimizer, leniter):


    num_training_steps = args.max_epochs * leniter

    if args.lrschedule == 'rop':
        lr_sch = ReduceLROnPlateau(optimizer, mode=args.mode,
                                    factor=args.gamma,
                                    patience=args.patience,
                                    threshold=args.threshold,
                                    threshold_mode=args.threshold_mode,
                                    min_lr=args.min_lr,
                                    eps=args.eps)
    elif args.lrschedule == 'cos':
        lr_sch = get_cosine_schedule_with_warmup(optimizer,
                                                ws,
                                                num_training_steps,
                                                last_epoch=args.pretrained_ep * leniter if args.continue_training else -1)
    elif args.lrschedule == 'sharp_cos':
        lr_sch = get_sharp_cosine_schedule_with_warmup(optimizer,
                                                ws,
                                                num_training_steps,
                                                last_epoch=args.pretrained_ep * leniter if args.continue_training else -1)

    elif args.lrschedule == 'lin':
        lr_sch = get_linear_schedule_with_warmup(optimizer,
                                                ws,
                                                num_training_steps,
                                                last_epoch=args.pretrained_ep * leniter if args.continue_training else -1)
    else:
        exit(f"check --lrschedule option!: {args.lrschedule}")

    return lr_sch



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


## openai gpt used warmup =2000, maxlr, minlr = 2.5e-4, 0 w/ cosine annealing
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_sharp_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.25, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - 3*num_training_steps + 2*num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., (1. + math.cos(2*math.pi * float(num_cycles)  * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

''' from pytorch docs: example

>>> # Assuming optimizer has two groups.
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95 ** epoch
>>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()
'''
