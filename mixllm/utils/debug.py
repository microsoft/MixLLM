# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

import torch
import gc
import numpy as np
import time

# TODO: remove this file.


def dump_mem(info: str = None):
    print(
        f"Torch used CUDA mem (info: {info}):\n"
        f"\tallocated: {torch.cuda.memory_allocated(0)/1024/1024/1024} GB\n"
        f"\treserved: {torch.cuda.memory_reserved(0)/1024/1024/1024} GB\n"
        f"\tmax reserved: {torch.cuda.max_memory_reserved(0)/1024/1024/1024} GB\n"
    )


def tensor_size_gigabytes(tensor):
    return tensor.nbytes / 1024 / 1024 / 1024


def dump_all_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and
                                        torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


# A helper class to print once. It will use a class attribute to record the printed status.
# Each call of the print function will be attached with a key. If the key is not in the global
# variable, then the print function will print the message and add the key to the global variable.
# Otherwise, the print function will not print the message.
class PrintOnce(object):
    printed = set()

    @classmethod
    def print(cls, msg: str, key: str):
        assert key is not None
        if key not in cls.printed:
            print(msg)
            cls.printed.add(key)


# Plot the 3D salience graph with `plot_surface` function. The input must be a 2D array.
def plot_2d_salience(salience_2d: map,
                     save_dir: str,
                     type: str = 'png',
                     all_in_one: bool = True):
    import matplotlib.pyplot as plt
    from pathlib import Path
    if all_in_one:
        ncols = min(7, len(salience_2d))
        nrows = int(np.ceil(len(salience_2d) / ncols))

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 subplot_kw=dict(projection="3d"),
                                 sharex=False,
                                 sharey=False,
                                 figsize=(ncols * 10, nrows * 10),
                                 dpi=100)
        fig.tight_layout(h_pad=2)

        items_list = list(salience_2d.items())
        for i, j in np.ndindex(axes.shape):
            idx = i * ncols + j
            if idx >= len(salience_2d):
                axes[i][j].remove()
                axes[i][j] = None
            else:
                items = items_list[idx]
                name = items[0]
                data = items[1]
                assert data.dim() == 2
                out_channel_range = np.arange(0, data.shape[0])
                in_channel_range = np.arange(0, data.shape[1])
                x, y = np.meshgrid(out_channel_range,
                                   in_channel_range,
                                   indexing='ij')

                ax = axes[i][j]
                ax.plot_surface(x,
                                y,
                                data.cpu().numpy(),
                                cmap=plt.cm.jet,
                                linewidth=0.1)
                ax.set_title(f'{name} Max:{data.max().item():.2f}')
                ax.set_xlabel('Out Channel')
                ax.set_ylabel('In Channel')
                ax.set_zlabel('Salience')

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        name = f'salience_{nrows}x{ncols}'
        plt.savefig(f'{save_dir}/{name}.{type}', format=type)
        plt.close()
    else:
        for ax, salience in zip(axes, salience_2d.items()):
            name = salience[0]
            data = salience[1]
            assert data.dim() == 2
            out_channel_range = np.arange(0, data.shape[0])
            in_channel_range = np.arange(0, data.shape[1])
            x, y = np.meshgrid(out_channel_range,
                               in_channel_range,
                               indexing='ij')

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(x, y, data, cmap=plt.cm.jet, linewidth=0.1)

            ax.set_xlabel('Out Channel')
            ax.set_ylabel('In Channel')
            ax.set_zlabel('Salience')
            ax.set_title(
                f'{name}, [{data.min().item():.2f}, {data.max().item():.2f}]')

            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{save_dir}/{name}.{type}', format=type)
            plt.close()


# Plot the 2D salience graph with `xxx` function. The input must be a 2D array.
# It applies the reduction on the specified direction to present the data.
def plot_reduced_salience(salience_2d: map,
                          save_dir: str,
                          type: str = 'png',
                          all_in_one: bool = True,
                          mean_only: bool = False,
                          peak_only: bool = False,
                          direction: str = 'both'):
    import matplotlib.pyplot as plt
    from pathlib import Path
    if mean_only and peak_only:
        raise ValueError(
            "Both mean_only and peak_only cannot be True at the same time.")
    if direction not in ['out_channel', 'in_channel', 'both']:
        raise ValueError(
            "The direction must be either 'out_channel', 'in_channel' or 'both'."
        )

    if all_in_one:
        # `7` is the linear operator number in each Llama decoder layer.
        ncols = min(7, len(salience_2d))
        nrows = int(np.ceil(len(salience_2d) / ncols))

        if direction == 'both':
            nrows *= 2

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex=False,
                                 sharey=False,
                                 figsize=(ncols * 10, nrows * 5),
                                 dpi=100)
        fig.tight_layout(h_pad=4, w_pad=2)

        items_list = list(salience_2d.items())
        for i, j in np.ndindex(axes.shape):
            idx = int(i / 2) * ncols + j
            if idx >= len(salience_2d):
                axes[i][j].remove()
                axes[i][j] = None
            else:
                items = items_list[idx]
                name = items[0]
                data = items[1]
                assert data.dim() == 2

                if direction == 'both':
                    if i % 2 == 0:
                        curr_direction = 'out_channel'
                    else:
                        curr_direction = 'in_channel'
                else:
                    curr_direction = direction

                if curr_direction == 'out_channel':
                    avg = data.mean(dim=1)
                    peak = data.max(dim=1).values
                elif curr_direction == 'in_channel':
                    avg = data.mean(dim=0)
                    peak = data.max(dim=0).values
                else:
                    raise NotImplementedError
                index_range = np.arange(0, avg.shape[0])
                avg = avg.cpu().numpy()
                peak = peak.cpu().numpy()

                ax = axes[i][j]
                if not peak_only:
                    ax.scatter(index_range,
                               avg,
                               c='b',
                               marker='o',
                               s=1,
                               label='avg')
                if not mean_only:
                    ax.scatter(index_range,
                               peak,
                               c='r',
                               marker='x',
                               s=1,
                               label='peak')
                ax.set_title(f'{name}')
                ax.set_xlabel(f'{curr_direction}')
                ax.set_ylabel('Reduced salience')

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        name = f'salience_{nrows}x{ncols}_mean-{mean_only}_peak-{peak_only}_direction-{direction}'
        plt.savefig(f'{save_dir}/{name}.{type}', format=type)
        plt.close()
    else:
        raise NotImplementedError
        for ax, salience in zip(axes, salience_2d.items()):
            name = salience[0]
            data = salience[1]
            assert data.dim() == 2

            if direction == 'out_channel':
                avg = data.mean(dim=1)
                peak = data.max(dim=1)
            elif direction == 'in_channel':
                avg = data.mean(dim=0)
                peak = data.max(dim=0)
            else:
                raise NotImplementedError
            index_range = np.arange(0, avg.shape[0])
            avg = avg.cpu().numpy()
            peak = peak.cpu().numpy()

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(index_range, avg, c='b', marker='o', s=1, label='avg')
            ax.scatter(index_range, peak, c='r', marker='x', s=1, label='peak')
            ax.set_title(f'{name}')
            ax.set_xlabel(f'{direction}')
            ax.set_ylabel('Reduced salience')

            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{save_dir}/{name}.{type}', format=type)
            plt.close()


# Save tensor to specific dir and file.
def save_tensor(tensor: torch.Tensor, save_dir: str, save_name: str):
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(tensor, f'{save_dir}/{save_name}')


# Load tensor from specific dir and file.
def load_tensor(dir: str, file_name: str):
    return torch.load(f'{dir}/{file_name}')


class TimeUtils:

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def record_and_update(self, msg: str):
        current_time = time.time()
        print(f"[{msg}]: {current_time - self.start_time}")
        self.start_time = current_time
