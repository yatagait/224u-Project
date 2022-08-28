import argparse
import random

# import ipdb
import numpy as np
import torch
import os
from data_generator import AmazonReview
from protonet import Protonet
parser = argparse.ArgumentParser(description='args')
parser.add_argument('--datasource', default='amazonreview', type=str,
                    help='amazonreview')
parser.add_argument('--select_data', default=-1, type=int, help='-1,0,1,2,3')
parser.add_argument('--test_dataset', default=-1, type=int)
parser.add_argument('--num_classes', default=2, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

## Training options
parser.add_argument('--metatrain_iterations', default=3000, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', default=25, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.001, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=2e-5, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--num_updates_test', default=10, type=int, help='num_updates in maml')
parser.add_argument('--update_batch_size', default=10, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=10, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')
parser.add_argument('--mix', default=False, action='store_true', help='use mixup or not')

## Logging, saving, and testing options
parser.add_argument('--log', default=1, type=int, help='if false, do not log summaries, for debugging code.')
parser.add_argument('--logdir', default='/iris/u/huaxiu/KGMeta/EMNLP_kgmeta_logs', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='/iris/u/yatagait/Data', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
parser.add_argument('--test_set', default=1, type=int,
                    help='Set to true to test on the the test set, False for the validation set.')
parser.add_argument('--trail', default=0, type=int, help='trail for each layer')
parser.add_argument('--warm_epoch', default=0, type=int, help='warm start epoch')
parser.add_argument('--ratio', default=1.0, type=float, help='warm start epoch')
parser.add_argument('--temp_scaling', default=1.0, type=float, help='temp for final softmax')
parser.add_argument('--trial', default=0, type=int, help='trial')
parser.add_argument('--all_data', default=False, action='store_true', help='use entire amazon or not')
parser.add_argument('--bert_model', default="albert-base-v1", type=str, help='bert model config')



args = parser.parse_args()
print(args)

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

random.seed(1)
np.random.seed(2)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

if args.all_data:
    args.datasource += '_all_data'

exp_string = 'ProtoNet' + '.data_' + str(args.datasource) + '.cls_' + str(args.num_classes) + '.mbs_' + str(
    args.meta_batch_size) + '.ubs_' + str(
    args.update_batch_size) + '.metalr' + str(args.meta_lr) + '.innerlr' + str(args.update_lr) + '.numupdates' + str(args.num_updates)

if args.num_filters != 64:
    exp_string += '.hidden' + str(args.num_filters)
if args.trail > 0:
    exp_string += '.trail{}'.format(args.trail)
if args.use_kg:
    exp_string += '.kg'
if args.mix:
    exp_string += '.mix'
    if args.mix_beta != 2.0:
        exp_string += '.b{}'.format(args.mix_beta)
if args.task_calibration:
    exp_string += '.task_calibration'
if args.trial > 0:
    exp_string += '.trial{}'.format(args.trial)
if args.bert_model != "albert-base-v1":
    exp_string += '.{}'.format(args.bert_model)

print(exp_string)


def train(args, protonet, optimiser):
    Print_Iter = 100
    Save_Iter = 200
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    if 'amazonreview' in args.datasource:
        train_dataset = AmazonReview(args, 'train', bert_model=args.bert_model)
        indomain_val_dataset = AmazonReview(args, 'indomain-val', bert_model=args.bert_model)
        val_dataset = AmazonReview(args, 'val', bert_model=args.bert_model)

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_dataset):
        protonet.train()
        if step > args.metatrain_iterations:
            break
        task_losses = []
        task_acc = []
        for meta_batch in range(args.meta_batch_size):
            if args.mix:
                loss_val, acc_val = protonet.forward_metamix(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch], y_qry[meta_batch])
            else:
                # Default
                dists, yq = protonet(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch], y_qry[meta_batch])
                if args.task_calibration:
                    mean_features = dists.view(protonet.args.num_classes, protonet.args.update_batch_size_eval, -1).mean(1)
                    assert len(mean_features) == args.num_classes
                loss_val, acc_val = protonet.loss(dists, yq)
            task_losses.append(loss_val)
            task_acc.append(acc_val)

        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_acc = torch.stack(task_acc).mean()

        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

        if step != 0 and step % Print_Iter == 0:
            print(
                'iter: {}, loss_all: {}, acc: {}'.format(
                    step, print_loss, print_acc))
            # Run Results columns:
            result_string = 'RunResults:ProtoNet,{},{},{},4,{},{},{}'.format(args.update_batch_size,
                                                                  args.num_classes,
                                                                  args.num_filters,
                                                                  args.datasource,
                                                                  'Train',
                                                                  step)
            if args.mix:
                result_string += ",1"
            else:
                result_string += ",0"
            result_string += ",{:.6f}".format(print_acc)
            print(result_string)
            print_loss, print_acc = 0.0, 0.0
            # Get Indomain-Val set Accuracy
            test(args, protonet, step, indomain_val_dataset, type='indomain-val')
            # Get Outdomain-Val Set Accuracy
            test(args, protonet, step, val_dataset,type='val')
        else:
            print_loss += meta_batch_loss / Print_Iter
            print_acc += meta_batch_acc / Print_Iter

        if step != 0 and step % Save_Iter == 0:
            torch.save(protonet.state_dict(),
                       '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))


def test(args, protonet, test_epoch, dataset, type='test'):
    protonet.eval()
    res_acc = []
    args.meta_batch_size = 1

    logits_list = []
    labels_list = []

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataset):
        if step > 300:
            break
        with torch.no_grad():
            dists, yq = protonet(x_spt[0], y_spt[0], x_qry[0], y_qry[0])
            _, acc_val = protonet.loss(dists, yq)
            logits_list.append(dists.cpu())
            labels_list.append(yq.cpu())
            res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)
    dists = torch.cat(logits_list)
    print('{}_epoch is {}, acc is {}, ci95 is {}'.format(type, test_epoch, np.mean(res_acc),
                                                           1.96 * np.std(res_acc) / np.sqrt(
                                                               args.num_test_task * args.meta_batch_size)))
    # Run Results columns:
    result_string = 'RunResults:ProtoNet,{},{},{},4,{},{},{}'.format(args.update_batch_size,
                                                                  args.num_classes,
                                                                  args.num_filters,
                                                                  args.datasource,
                                                                  type,
                                                                  test_epoch)
    if args.mix:
        result_string += ",1"
    else:
        result_string += ",0"
    result_string += ",{:.6f}".format(np.mean(res_acc))
    print(result_string)

def main():
    protonet = Protonet(args).cuda()

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print(model_file)
        protonet.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.AdamW(list(protonet.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.train == 1:
        train(args, protonet, meta_optimiser)
    else:
        if 'amazonreview' in args.datasource:
            val_dataset = AmazonReview(args, 'val', bert_model=args.bert_model)
            test_dataset = AmazonReview(args, 'test', bert_model=args.bert_model)
        test_epoch = args.test_epoch
        try:
            model_file = '{0}/{2}/model{1}'.format(args.logdir, test_epoch, exp_string)
            protonet.load_state_dict(torch.load(model_file))

            test(args, protonet, test_epoch, val_dataset, type='val')
            test(args, protonet, test_epoch, test_dataset, type='test')
        except IOError:
            # continue
            print("error")


if __name__ == '__main__':
    main()
