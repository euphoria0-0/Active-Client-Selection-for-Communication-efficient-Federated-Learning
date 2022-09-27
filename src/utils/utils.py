import platform
import time
import os


def save_files(args):
    args.machine = platform.uname().node
    args.start = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    #args.save_path = f'./results/{args.dataset}/{args.method}-{args.fed_algo}-{args.num_clients_per_round}-{args.total_num_clients}/{args.start}'
    alpha = f'_a{args.alpha2}' if 'MaxEntropySampling' in args.method else ''
    dirichlet_alpha = f'_Da{args.dirichlet_alpha}' if args.dataset == 'PartitionedCIFAR10' else ''
    add = ''
    if args.loss_div_sqrt:
        add += '_sqrt'
    elif args.loss_sum:
        add += '_total'
    path = f'./results/{args.dataset}/{args.method}{alpha}{dirichlet_alpha}{add}-{args.start}'
    os.makedirs(path, exist_ok=True)
    if args.loss_sum:
        args.comment += '_total'
    elif args.loss_div_sqrt:
        args.comment += '_sqrt'
    args.file_name_opt = f'{args.method}-{args.fed_algo}-{args.num_clients_per_round}-{args.total_num_clients}{args.comment}'

    opts_file = open(f'{path}/options_{args.file_name_opt}_{args.start}.txt', 'w')
    opts_file.write('=' * 30 + '\n')
    for arg in vars(args):
        opts_file.write(f' {arg} = {getattr(args, arg)}\n')
    opts_file.write('=' * 30 + '\n')

    result_files = {}
    result_files['result'] = open(f'{path}/results_{args.file_name_opt}_{args.start}.txt', 'w')
    result_files['result'].write('Round,TrainLoss,TrainAcc,TestLoss,TestAcc\n')

    result_files['client'] = open(f'{path}/client_{args.file_name_opt}_{args.start}.txt', 'w')

    if args.save_probs:
        result_files['prob'] = open(f'{path}/probs_{args.file_name_opt}_{args.start}.txt', 'w')
        result_files['num_samples'] = open(f'{path}/num_samples_{args.file_name_opt}_{args.start}.txt', 'w')

    return result_files