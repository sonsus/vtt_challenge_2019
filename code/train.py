from ignite.engine.engine import Engine, State, Events

from ckpt import get_model_ckpt, save_ckpt
from loss import get_loss
from optimizer import get_optimizer

from utils import *
from lrschedulers import *
from metric import get_metrics
from evaluate import get_evaluator, evaluate_once
from metric.stat_metric import StatMetric
from munch import Munch as M
from logger import *
from tqdm import tqdm
from ignite.handlers import EarlyStopping
from functools import partial


def get_trainer(args, model, loss_fn, optimizer):
    def update_model(trainer, batch):
        model.train()
        optimizer.zero_grad()
        net_inputs, target = prepare_batch(args, batch, model.vocab)
        net_inputs = M(net_inputs) # que, images, answers, subtitle,
        y_pred = model(**net_inputs)
        batch_size = y_pred.shape[0]
        loss, stats = loss_fn(y_pred, target)
        loss.backward()
        optimizer.step()
        return loss.item(), stats, batch_size, y_pred.detach(), target.detach()

    trainer = Engine(update_model)

    metrics = {
        'loss': StatMetric(output_transform=lambda x: (x[0], x[2])),
        'top1_acc': StatMetric(output_transform=lambda x: ((x[3].argmax(dim=-1) == x[4]).float().mean().item(), x[2]))
    }
    if hasattr(loss_fn, 'get_metric'):
        metrics = {**metrics, **loss_fn.get_metric()}

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def train(args):
    args, model, iters, vocab, ckpt_available = get_model_ckpt(args)

    if ckpt_available:
        print("loaded checkpoint {}".format(args.ckpt_name))
    loss_fn = get_loss(args, vocab)
    optimizer = get_optimizer(args, model)

    len_train = len(iters['train'].it) # iters['train'].it : text iterator
                                        # iters['train']: paired object of image_iterator and text_iterator
    lrsch = get_scheduler(args, optimizer, len_train)

    trainer = get_trainer(args, model, loss_fn, optimizer)
    metrics = get_metrics(args, vocab)
    evaluator = get_evaluator(args, model, loss_fn, metrics)

    logger = get_logger(args)


    progress_bar = tqdm(total=args.max_epochs * len_train)


    def see_acc(engine):
        return engine.state.metrics['top1_acc']
    earlystop_handler = EarlyStopping(patience = args.patience+10, score_function=see_acc, trainer=trainer )
    evaluator.add_event_handler(Events.COMPLETED, earlystop_handler)

    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        print("Begin Training")

    @trainer.on(Events.ITERATION_STARTED)
    def log_lr_iter(engine):
        log_lr(logger, 'lr/iter', optimizer, engine.state.iteration)
    @trainer.on(Events.EPOCH_STARTED)
    def log_lr_ep(engine):
        log_lr(logger, 'lr/ep', optimizer, engine.state.epoch)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iter_results(engine):
        log_results(logger, 'train/iter', engine.state, engine.state.iteration)
        progress_bar.update()
        if args.lrschedule != 'rop':
            lrsch.step()


    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_epoch(engine):
        log_results(logger, 'train/epoch', engine.state, engine.state.epoch)
        state = evaluate_once(evaluator, iterator=iters['val'])


        log_results(logger, 'valid/epoch', state, engine.state.epoch)
        if args.lrschedule == 'rop':
            lrsch.step(state.metrics['loss'])
        save_ckpt(args, engine.state.epoch, state.metrics['loss'], model, vocab) # save by val loss


    trainer.run(iters['train'], max_epochs=args.max_epochs)
    progress_bar.close()
