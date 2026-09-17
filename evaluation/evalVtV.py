import argparse
import json
import math
import re
from pathlib import Path

from sklearn.metrics import precision_recall_curve, f1_score

from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata
def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
        
import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def get_frame_scores(pred, gt, mask):
    return gt[mask == 1], pred[mask == 1]


def compute_kendall(pred, gt, mask, num_repeat=10):
    pred_np = pred.sigmoid().cpu().numpy()
    gt_np = gt.cpu().numpy()
    mask_np = mask.cpu().numpy()
    f = lambda x, y: kendalltau(rankdata(-x), rankdata(-y))

    bs = pred_np.shape[0]
    kendall_scores = np.zeros(bs)

    for i in range(bs):
        gt_i, pred_i = get_frame_scores(pred_np[i], gt_np[i], mask_np[i])
        kendall_tau = f(gt_i, pred_i)[0]
        if math.isnan(kendall_tau):
            kendall_scores[i] = 0.0
        else:
            kendall_scores[i] = kendall_tau

    kendall_scores = kendall_scores.reshape(-1, num_repeat)
    kendall_min = np.min(kendall_scores, axis=1)
    kendall_max = np.max(kendall_scores, axis=1)
    kendall_mean = np.mean(kendall_scores, axis=1)

    return torch.from_numpy(kendall_min), torch.from_numpy(kendall_max), torch.from_numpy(kendall_mean)


def compute_spearman(pred, gt, mask, num_repeat=10):
    pred_np = pred.sigmoid().cpu().numpy()
    gt_np = gt.cpu().numpy()
    mask_np = mask.cpu().numpy()
    f = lambda x, y: spearmanr(x, y)

    bs = pred_np.shape[0]
    spearman_scores = np.zeros(bs)

    for i in range(bs):
        gt_i, pred_i = get_frame_scores(pred_np[i], gt_np[i], mask_np[i])
        spearman_rho = f(gt_i, pred_i)[0]
        if math.isnan(spearman_rho):
            spearman_scores[i] = 0.0
        else:
            spearman_scores[i] = spearman_rho

    spearman_scores = spearman_scores.reshape(-1, num_repeat)
    spearman_min = np.min(spearman_scores, axis=1)
    spearman_max = np.max(spearman_scores, axis=1)
    spearman_mean = np.mean(spearman_scores, axis=1)

    return torch.from_numpy(spearman_min), torch.from_numpy(spearman_max), torch.from_numpy(spearman_mean)


def compute_f1(pred, gt, mask, num_repeat=10):
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()
    mask_np = mask.cpu().numpy()

    bs = pred_np.shape[0]
    f1 = np.zeros(bs)

    for i in range(bs):
        gt_i, pred_i = get_top15_frames(pred_np[i], gt_np[i], mask_np[i])
        f1[i] = f1_score(gt_i, pred_i)

    f1 = f1.reshape(-1, num_repeat)
    f1_min = np.min(f1, axis=1)
    f1_max = np.max(f1, axis=1)
    f1_mean = np.mean(f1, axis=1)

    return torch.from_numpy(f1_min), torch.from_numpy(f1_max), torch.from_numpy(f1_mean)


def get_top15_frames(pred, gt, mask):
    _pred = pred[mask == 1]
    n_top15 = int(len(_pred) * 0.15)
    val_top15 = np.sort(_pred)[::-1][int(n_top15)]
    pred_top15 = (_pred >= val_top15).astype(np.float32)

    return gt[mask == 1], pred_top15


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

import builtins as __builtin__
from datetime import timedelta
def setup_for_distributed(is_master, fpath):
    """
    This function disables printing when not in master process
    """
    builtin_print = __builtin__.print
    start_time = time.time()
    fpath = 'output.txt' if fpath is None else fpath

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            # now = datetime.datetime.now().time()
            now = time.time()
            elapsed_seconds = round(now - start_time)
            f = open(fpath, "a+")
            builtin_print('[{} - {}] '.format(time.strftime("%x %X"), timedelta(seconds=elapsed_seconds), ),
                          end='')  # print with time stamp
            builtin_print(*args, **kwargs)
            builtin_print('[{} - {}] '.format(time.strftime("%x %X"), timedelta(seconds=elapsed_seconds), ), end='',
                          file=f)  # print with time stamp
            builtin_print(*args, **kwargs, file=f)
            f.close()

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0, args.logger_pth)


def update_config(config, args):
    for k, v in vars(args).items():
        if k in config:
            config[k] = v
    return config


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def load_vsum_predictions(predictions_path, index_base=1):
    """Legge i JSONL di predizioni: {video_id: (indici o stringa V2Xum, base degli indici)}."""
    predictions_path = Path(predictions_path)
    files = sorted(predictions_path.glob("*.jsonl")) if predictions_path.is_dir() else [predictions_path]
    if not files:
        raise FileNotFoundError(f"Nessuna predizione in {predictions_path}")
    predictions = {}
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("record_type") not in (None, "sample") or "video_id" not in row:
                    continue
                video_id = str(row["video_id"])
                if video_id in predictions:
                    raise ValueError(f"video_id duplicato: {video_id}")
                if "selected_frame_indices" in row:
                    predictions[video_id] = ([int(x) for x in row["selected_frame_indices"]], 0)
                else:
                    predictions[video_id] = (str(row.get("summary", "")), index_base)
    return predictions


def load_vsum_annotations(annotations_path):
    """Legge le annotazioni VideoXum (lista o dict) indicizzate per video_id."""
    with Path(annotations_path).open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    if isinstance(annotations, list):
        annotations_by_video = {}
        for position, row in enumerate(annotations):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Annotazione in posizione {position} non valida: atteso un oggetto JSON"
                )
            video_id = row.get("video_id") or row.get("id")
            if not video_id:
                raise ValueError(
                    f"Annotazione in posizione {position} senza video_id o id"
                )
            video_id = str(video_id)
            if video_id in annotations_by_video:
                raise ValueError(f"video_id duplicato nelle annotazioni: {video_id}")
            annotations_by_video[video_id] = row
        annotations = annotations_by_video
    elif not isinstance(annotations, dict):
        raise ValueError(
            "Formato annotazioni non valido: attesa una lista o un oggetto JSON"
        )
    return annotations


def annotation_references(video_id, row):
    """Restituisce (frame_count, matrice binaria 10 x frame_count) di un video."""
    required_fields = {"sampled_frames", "vsum_onehot"}
    missing_fields = sorted(required_fields - row.keys())
    if missing_fields:
        raise ValueError(
            f"{video_id}: annotazione frame-level non valida; campi "
            f"mancanti: {', '.join(missing_fields)}. Usa il metadata "
            "VideoXum con sampled_frames e vsum_onehot."
        )
    frame_count = int(row["sampled_frames"])
    references = np.asarray(row["vsum_onehot"], dtype=np.int8)
    if references.shape != (10, frame_count):
        raise ValueError(f"{video_id}: shape {references.shape}, attesa (10, {frame_count})")
    return frame_count, references


def prediction_mask(prediction_data, item_base, frame_count):
    """Converte una predizione in maschera binaria; restituisce (maschera, indici fuori range)."""
    if isinstance(prediction_data, list):
        raw = prediction_data
    else:
        # V2Xum contiene numeri con zero iniziale (es. 06).
        raw = [int(x) for x in re.findall(r"\d+", prediction_data)]
    indices = [x - item_base for x in raw]
    invalid = [x + item_base for x in indices if not 0 <= x < frame_count]
    prediction = np.zeros(frame_count, dtype=np.int8)
    valid = {x for x in indices if 0 <= x < frame_count}
    if valid:
        prediction[list(valid)] = 1
    return prediction, invalid


def evaluate_vsum_files(predictions_path, annotations_path, output_path, index_base=1):
    """Valuta shard V2Xum JSONL contro le 10 annotazioni binarie VideoXum."""
    predictions = load_vsum_predictions(predictions_path, index_base)
    annotations = load_vsum_annotations(annotations_path)
    names = ("f1_min", "f1_max", "f1_mean", "kendall", "spearman")
    collected = {name: [] for name in names}
    missing = invalid_total = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for video_id, row in annotations.items():
            if video_id not in predictions:
                missing += 1
                continue
            frame_count, references = annotation_references(video_id, row)
            prediction, invalid = prediction_mask(*predictions[video_id], frame_count)
            invalid_total += len(invalid)

            intersections = (references * prediction[None, :]).sum(axis=1)
            denominators = references.sum(axis=1) + prediction.sum()
            f1s = np.divide(2.0 * intersections, denominators,
                            out=np.zeros(10, dtype=float), where=denominators != 0)
            saliency = references.mean(axis=0)
            tau = kendalltau(rankdata(-prediction), rankdata(-saliency))[0]
            rho = spearmanr(prediction, saliency)[0]
            metrics = {
                "f1_min": float(f1s.min()), "f1_max": float(f1s.max()),
                "f1_mean": float(f1s.mean()),
                "kendall": 0.0 if math.isnan(tau) else float(tau),
                "spearman": 0.0 if math.isnan(rho) else float(rho),
            }
            for name in names:
                collected[name].append(metrics[name])
            output.write(json.dumps({
                "record_type": "sample", "video_id": video_id,
                "selected_frames": int(prediction.sum()), "invalid_indices": invalid,
                "f1_per_annotator": f1s.tolist(), **metrics,
            }) + "\n")

        evaluated = len(collected["f1_mean"])
        if not evaluated:
            raise ValueError("Nessuna predizione corrisponde alle annotazioni")
        summary = {
            "record_type": "summary", "videos_evaluated": evaluated,
            "predictions_total": len(predictions), "annotations_total": len(annotations),
            "missing_predictions": missing,
            "extra_predictions": len(set(predictions) - set(annotations)),
            "invalid_indices": invalid_total, "index_base": index_base,
            **{name: float(np.mean(values)) for name, values in collected.items()},
        }
        output.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Risultati salvati in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuta predizioni V-sum su VideoXum")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=1)
    cli = parser.parse_args()
    evaluate_vsum_files(cli.predictions, cli.annotations, cli.output, cli.index_base)
