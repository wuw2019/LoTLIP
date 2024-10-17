import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from training.precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

def run_long_retrieval(model, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    image_features = []
    text_features = []
    with torch.no_grad():
        for images, texts in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            texts = texts.to(args.device)

            with autocast():
                # predict
                model_out = model(images, texts)
                image_features.append(model_out["image_features"])
                text_features.append(model_out["text_features"])

        image_features = torch.cat(image_features, dim=0)
        text_features = torch.cat(text_features, dim=0)
        
        t2i_similarity = text_features@image_features.t() 
        _, preds = t2i_similarity.max(dim=1)
        pred_true = torch.sum(preds==torch.arange(text_features.shape[0]).to(args.device))
        t2i_acc = float(pred_true/text_features.shape[0])

        i2t_similarity = t2i_similarity.t() 
        _, preds = i2t_similarity.max(dim=1)
        pred_true = torch.sum(preds==torch.arange(image_features.shape[0]).to(args.device))
        i2t_acc = float(pred_true/image_features.shape[0])

        return t2i_acc, i2t_acc


def zero_shot_eval(model, data, epoch, args, tokenizer=None):

    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    results = {}

    if 'imagenet-val' in data or 'imagenet-v2' in data:
        logging.info('Starting zero-shot imagenet.')
        if tokenizer is None:
            tokenizer = get_tokenizer(args.model)
        
        logging.info('Building zero-shot classifier')
        autocast = get_autocast(args.precision)
        with autocast():
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        logging.info('Using classifier')
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    if 'share4v-retrieval' in data:
        logging.info('Starting zero-shot share4v retrieval.')
        for key in data.keys():
            if 'share4v-retrieval' in key:
                num_share4v_val_data = data[key].dataloader.dataset.total_len
                t2i_acc, i2t_acc = run_long_retrieval(model, data[key].dataloader, args)
                results['share4v-retrieval-'+str(num_share4v_val_data)+'-i2t-top1'] = i2t_acc
                results['share4v-retrieval-'+str(num_share4v_val_data)+'-t2i-top1'] = t2i_acc
        logging.info('Finished zero-shot share4v retrieval.')

                
    if 'dci-retrieval' in data:
        logging.info('Starting zero-shot dci retrieval.')
        t2i_acc, i2t_acc = run_long_retrieval(model, data['dci-retrieval'].dataloader, args)
        results['dci-retrieval-i2t-top1'] = i2t_acc
        results['dci-retrieval-t2i-top1'] = t2i_acc
        logging.info('Finished zero-shot dci retrieval.')

    if 'iiw-retrieval' in data:
        logging.info('Starting zero-shot iiw retrieval.')
        t2i_acc, i2t_acc = run_long_retrieval(model, data['iiw-retrieval'].dataloader, args)
        results['iiw-retrieval-i2t-top1'] = i2t_acc
        results['iiw-retrieval-t2i-top1'] = t2i_acc

        logging.info('Finished zero-shot iiw retrieval.')

    return results
