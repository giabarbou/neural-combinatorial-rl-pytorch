import argparse
import os
from tqdm import tqdm
import pprint as pp
import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value
import time

from neural_combinatorial_rl import NeuralCombOptRL
from plot_attention import plot_attention
import csv

print('Using pytorch ', torch.__version__)


def str2bool(v):
    return v.lower() in ('true', '1')


def append2csv(fname, values):
    with open(fname, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Combinatorial Optimization with RL")

    # Data
    parser.add_argument('--task', default='tsp_10', help="The task to solve, in the form {COP}_{size}, e.g., tsp_20")
    parser.add_argument('--batch_size', default=128, help='')
    parser.add_argument('--val_batch_size', default=128, help='')
    parser.add_argument('--train_size', default=12800, help='')
    parser.add_argument('--val_size', default=1280, help='')
    # Network
    parser.add_argument('--embedding_dim', default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', default=128, help='Dimension of hidd en layers in Enc/Dec')
    parser.add_argument('--n_process_blocks', default=3,
                        help='Number of process block iters to run in the Critic network')
    parser.add_argument('--n_glimpses', default=1, help='No. of glimpses to use in the pointer network')
    parser.add_argument('--use_tanh', type=str2bool, default=True)
    parser.add_argument('--tanh_exploration', default=10,
                        help='Hyperparam controlling exploration in the pointer net by scaling the tanh in the softmax')
    parser.add_argument('--dropout', default=0., help='')
    parser.add_argument('--terminating_symbol', default='<0>', help='')
    parser.add_argument('--beam_size', default=1, help='Beam width for beam search')

    # Training
    parser.add_argument('--actor_net_lr', default=1e-3, help="Set the learning rate for the actor network")
    parser.add_argument('--actor_lr_decay_step', default=5000, help='')
    parser.add_argument('--actor_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--critic_beta', type=float, default=0.8, help='Exp mvg average decay')
    parser.add_argument('--critic_net_lr', default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--critic_lr_decay_step', default=5000, help='')
    parser.add_argument('--critic_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--reward_scale', default=1, type=float, help='')
    parser.add_argument('--is_train', type=str2bool, default=True, help='')
    parser.add_argument('--n_epochs', default=50, help='')
    parser.add_argument('--random_seed', default=1, help='')
    parser.add_argument('--max_grad_norm', default=1.0, help='Gradient clipping')
    parser.add_argument('--use_cuda', type=str2bool, default=True, help='')

    # Misc
    parser.add_argument('--log_step', default=500, help='Log info every log_step steps')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--disable_tensorboard', type=str2bool, default=True)
    parser.add_argument('--plot_attention', type=str2bool, default=False)
    parser.add_argument('--disable_progress_bar', type=str2bool, default=True)
    parser.add_argument('--results_dir', type=str, default='result_files')

    args = vars(parser.parse_args())

    # Pretty print the run args
    pp.pprint(args)

    # Set the random seed
    torch.manual_seed(int(args['random_seed']))

    # Optionally configure tensorboard
    if not args['disable_tensorboard']:
        configure(os.path.join(args['log_dir'], args['task'], args['run_name']))

    # Task specific configuration - generate dataset if needed
    task = args['task'].split('_')
    COP = task[0]
    size = int(task[1])
    data_dir = os.path.join('data', COP)

    constraints = []
    if COP == 'sort':
        import sorting_task

        input_dim = 1
        reward_fn = sorting_task.reward
        train_fname, val_fname = sorting_task.create_dataset(
            int(args['train_size']),
            int(args['val_size']),
            data_dir,
            data_len=size)
        training_dataset = sorting_task.SortingDataset(train_fname)
        val_dataset = sorting_task.SortingDataset(val_fname)

    elif COP == 'tsp':
        import tsp_task

        input_dim = 2
        reward_fn = tsp_task.reward
        val_fname = tsp_task.create_dataset(
            problem_size=str(size),
            data_dir=data_dir)
        training_dataset = tsp_task.TSPDataset(train=True, size=size,
             num_samples=int(args['train_size']))
        val_dataset = tsp_task.TSPDataset(train=True, size=size,
                num_samples=int(args['val_size']))

    elif COP == 'carpool':
        import carpool_task

        input_dim = 2
        reward_fn = carpool_task.reward
        train_fname = carpool_task.create_dataset(int(args['train_size']), num_points=size, data_dir=data_dir,
                                                  purpose='train')
        val_fname = carpool_task.create_dataset(int(args['val_size']), num_points=size, data_dir=data_dir,
                                                purpose='test')

        training_dataset = carpool_task.CarpoolDataset(train_fname)
        val_dataset = carpool_task.CarpoolDataset(val_fname)

        # making sure our model doesn't use the start, goal and threshold points
        constraints = [carpool_task.IDX_THR,
                       carpool_task.IDX_GOAL,
                       carpool_task.IDX_START]

        problem_size = size
        size = carpool_task.CAPACITY + len(constraints)

    else:
        print('Currently unsupported task!')
        exit(1)

    # Load the model parameters from a saved state
    if args['load_path'] != '':
        print('  [*] Loading model from {}'.format(args['load_path']))

        model = torch.load(
            os.path.join(
                os.getcwd(),
                args['load_path']
            ))
        model.actor_net.decoder.max_length = size
        model.is_train = args['is_train']
    else:

        # Instantiate the Neural Combinatorial Opt with RL module
        model = NeuralCombOptRL(
                                input_dim,
                                int(args['embedding_dim']),
                                int(args['hidden_dim']),
                                size,  # decoder len
                                args['terminating_symbol'],
                                int(args['n_glimpses']),
                                int(args['n_process_blocks']),
                                float(args['tanh_exploration']),
                                args['use_tanh'],
                                int(args['beam_size']),
                                reward_fn,
                                args['is_train'],
                                args['use_cuda'],
                                constraints)

    save_dir = os.path.join(os.getcwd(),
                            args['output_dir'],
                            args['task'],
                            args['run_name'])

    results_dir = os.path.join(os.getcwd(),
                              args['results_dir'],
                              args['task'])

    try:
        os.makedirs(save_dir)
        os.makedirs(results_dir)
    except:
        pass

    # critic_mse = torch.nn.MSELoss()
    # critic_optim = optim.Adam(model.critic_net.parameters(), lr=float(args['critic_net_lr']))
    actor_optim = optim.Adam(model.actor_net.parameters(), lr=float(args['actor_net_lr']))

    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
            range(int(args['actor_lr_decay_step']), int(args['actor_lr_decay_step']) * 1000,
                int(args['actor_lr_decay_step'])), gamma=float(args['actor_lr_decay_rate']))

    # critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
    #        range(int(args['critic_lr_decay_step']), int(args['critic_lr_decay_step']) * 1000,
    #            int(args['critic_lr_decay_step'])), gamma=float(args['critic_lr_decay_rate']))

    training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
                                     shuffle=True, num_workers=4)

    validation_dataloader = DataLoader(val_dataset, batch_size=int(args['val_batch_size']),
                                       shuffle=True, num_workers=1)

    critic_exp_mvg_avg = torch.zeros(1)
    beta = args['critic_beta']

    if args['use_cuda']:
        model = model.cuda()
        # critic_mse = critic_mse.cuda()
        critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

    step = 0
    val_step = 0

    if not args['is_train']:
        args['n_epochs'] = '1'

    overall_avg_reward_train = 0.0
    overall_avg_reward_val = 0.0
    overall_reward_var_train = 0.0
    overall_reward_var_val = 0.0
    training_time = 0.0
    validation_time = 0.0

    epoch = int(args['epoch_start'])
    for i in range(epoch, epoch + int(args['n_epochs'])):
        if args['is_train']:
            time_start = time.time()

            # put in train mode!
            model.train()

            avg_reward = []

            # sample_batch is [batch_size x input_dim x sourceL]
            for batch_id, sample_batch in enumerate(tqdm(training_dataloader, disable=args['disable_progress_bar'])):

                bat = Variable(sample_batch)
                if args['use_cuda']:
                    bat = bat.cuda()

                R, probs, actions, actions_idxs = model(bat)

                avg_reward.append(R[0].item())

                if batch_id == 0:
                    critic_exp_mvg_avg = R.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

                advantage = R - critic_exp_mvg_avg

                logprobs = 0
                nll = 0
                for prob in probs:
                    # compute the sum of the log probs
                    # for each tour in the batch
                    logprob = torch.log(prob)
                    nll += -logprob
                    logprobs += logprob

                # guard against nan
                nll[(nll != nll).detach()] = 0.
                # clamp any -inf's to 0 to throw away this tour
                logprobs[(logprobs < -1000).detach()] = 0.

                # multiply each time step by the advanrate
                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                actor_optim.zero_grad()

                actor_loss.backward()

                # clip gradient norms
                torch.nn.utils.clip_grad_norm_(
                    model.actor_net.parameters(),
                    float(args['max_grad_norm']), norm_type=2)

                actor_optim.step()
                actor_scheduler.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                # critic_scheduler.step()
                #
                # R = R.detach()
                # critic_loss = critic_mse(v.squeeze(1), R)
                # critic_optim.zero_grad()
                # critic_loss.backward()
                #
                # torch.nn.utils.clip_grad_norm_(model.critic_net.parameters(),
                #        float(args['max_grad_norm']), norm_type=2)
                #
                # critic_optim.step()

                step += 1

                if not args['disable_tensorboard']:
                    log_value('avg_reward', R.mean().item(), step)
                    log_value('actor_loss', actor_loss.item(), step)
                    # log_value('critic_loss', critic_loss.item(), step)
                    log_value('critic_exp_mvg_avg', critic_exp_mvg_avg.item(), step)
                    log_value('nll', nll.mean().item(), step)

                if step % int(args['log_step']) == 0:
                    print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(
                        i, batch_id, R.mean().item()))
                    example_output = []
                    example_input = []
                    for idx, action in enumerate(actions):
                        if task[0] == 'tsp' or task[0] == 'carpool':
                            example_output.append(actions_idxs[idx][0].item())
                        else:
                            example_output.append(action[0].item())  # <-- ??
                        example_input.append(sample_batch[0, :, idx][0])
                    # print('Example train input: {}'.format(example_input))
                    print('Example train output: {}'.format(example_output))

            overall_avg_reward_train = np.mean(avg_reward)
            overall_reward_var_train = np.var(avg_reward)
            training_time = time.time() - time_start
            print('Training overall avg_reward: {}'.format(overall_avg_reward_train))
            print('Training overall reward var: {}'.format(overall_reward_var_train))

        # Use beam search decoding for validation
        # model.actor_net.decoder.decode_type = "beam_search"

        print('\n~Validating~\n')

        time_start = time.time()

        example_input = []
        example_output = []
        avg_reward = []

        # put in test mode!
        model.eval()

        for batch_id, val_batch in enumerate(tqdm(validation_dataloader, disable=args['disable_progress_bar'])):
            bat = Variable(val_batch)

            if args['use_cuda']:
                bat = bat.cuda()

            R, probs, actions, action_idxs = model(bat)

            avg_reward.append(R[0].item())
            val_step += 1.

            if not args['disable_tensorboard']:
                log_value('val_avg_reward', R[0].item(), int(val_step))

            if val_step % int(args['log_step']) == 0:
                example_output = []
                example_input = []
                for idx, action in enumerate(actions):
                    if task[0] == 'tsp' or task[0] == 'carpool':
                        example_output.append(action_idxs[idx][0].item())
                    else:
                        example_output.append(action[0].item())
                    # example_input.append(bat[0, :, idx].item())
                    example_input.append(bat[0, :, idx][0])
                print('Step: {}'.format(batch_id))
                # print('Example test input: {}'.format(example_input))
                print('Example test output: {}'.format(example_output))
                print('Example test reward: {}'.format(R[0].item()))

                if args['plot_attention']:
                    probs = torch.cat(probs, 0)
                    plot_attention(example_input,
                            example_output, probs.data.cpu().numpy())

        overall_avg_reward_val = np.mean(avg_reward)
        overall_reward_var_val = np.var(avg_reward)
        validation_time = time.time() - time_start
        print('Validation overall avg_reward: {}'.format(overall_avg_reward_val))
        print('Validation overall reward var: {}'.format(overall_reward_var_val))

        if args['is_train']:
            model.actor_net.decoder.decode_type = "stochastic"

            print('Saving model...')

            torch.save(model, os.path.join(save_dir, 'epoch-{}.pt'.format(i)))

            results_list = [overall_avg_reward_train,
                            overall_avg_reward_val,
                            overall_reward_var_train,
                            overall_reward_var_val,
                            training_time,
                            validation_time]

            results_fname = os.path.join(results_dir, args['run_name'] + '.csv')
            append2csv(results_fname, results_list)

            # If the task requires generating new data after each epoch, do that here!
            if COP == 'tsp':
                training_dataset = tsp_task.TSPDataset(train=True, size=size,
                                                       num_samples=int(args['train_size']))

                training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
                                                 shuffle=True, num_workers=1)
            if COP == 'sort':
                train_fname, _ = sorting_task.create_dataset(
                                                            int(args['train_size']),
                                                            int(args['val_size']),
                                                            data_dir,
                                                            data_len=size)
                training_dataset = sorting_task.SortingDataset(train_fname)
                training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
                                                 shuffle=True, num_workers=1)
            # if COP == 'carpool':
            #     training_dataset = carpool_task.CarpoolDataset(train=True, size=problem_size,
            #                                                    num_samples=int(args['train_size']))
            #     training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
            #                                      shuffle=True, num_workers=1)
