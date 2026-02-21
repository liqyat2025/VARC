import argparse
from copy import deepcopy
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from utils.args import parse_args
from utils.distribution import init_distributed_mode
from utils.load_model import load_model_only, load_optimizer

from src.RARC_loader import build_dataloaders, IGNORE_INDEX
from utils.eval_utils_ttt import generate_predictions, get_eval_rot_transform_resolver


def _format_eta(seconds: float) -> str:
    total_seconds = int(max(seconds, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ttt_once(model, device, distributed, rank, train_loader, train_sampler, eval_loader, cur_attempt_idx):
    autocast_device_type = device.type if device.type in {"cuda", "cpu", "mps"} else "cuda"
    is_main_process = (not distributed) or rank == 0

    global_start = time.time()
    previous_total_steps = 0
    optimizer, scaler, scheduler = load_optimizer(
        model=model, args=args, device=device, distributed=distributed, rank=rank
    )
    try:
        for epoch in range(0, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            sample_count = 0
            total_batches = len(train_loader)
            epoch_start = time.time()
            train_exact = 0
            train_examples = 0

            for step, batch in enumerate(train_loader, 1):
                inputs = batch["inputs"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                task_ids = batch["task_ids"].to(device)

                optimizer.zero_grad(set_to_none=True)
                
                # Use automatic mixed precision
                with autocast(device_type=autocast_device_type, enabled=scaler.is_enabled()):
                    logits = model(inputs, task_ids, attention_mask=attention_mask)
                    num_colors = logits.size(1)
                    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
                    loss = F.cross_entropy(
                        logits_flat,
                        targets.view(-1),
                        ignore_index=IGNORE_INDEX,
                    )

                batch_size = inputs.size(0)
                predictions = logits.argmax(dim=1)
                for idx in range(batch_size):
                    target = targets[idx]
                    prediction = predictions[idx]
                    valid = target != IGNORE_INDEX
                    if valid.any():
                        is_exact = bool(torch.equal(prediction[valid], target[valid]))
                    else:
                        is_exact = False
                    train_exact += int(is_exact)
                    train_examples += 1

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * batch_size
                sample_count += batch_size

                if total_batches > 0 and is_main_process and step % 10 == 0:  # Update every 10 steps
                    elapsed = time.time() - epoch_start
                    avg_step_time = elapsed / step
                    steps_completed = previous_total_steps + step
                    total_steps = len(train_loader) * args.epochs
                    remaining_steps = total_steps - steps_completed
                    elapsed_global = time.time() - global_start
                    avg_time_per_step_global = elapsed_global / max(steps_completed, 1)
                    eta = remaining_steps * avg_time_per_step_global
                    bar_length = 30
                    progress_ratio = steps_completed / total_steps if total_steps else 0
                    filled = int(bar_length * progress_ratio)
                    bar = "#" * filled + "-" * (bar_length - filled)
                    progress = 100.0 * progress_ratio
                    sys.stdout.write(
                        f"\rEpoch {epoch} [{bar}] {progress:5.1f}% ETA {_format_eta(eta)}"
                    )
                    sys.stdout.flush()

            if total_batches > 0 and is_main_process:
                sys.stdout.write("\n")
            previous_total_steps += total_batches

            epoch_duration = time.time() - epoch_start if total_batches > 0 else 0.0

            train_totals = torch.tensor(
                [running_loss, sample_count, train_exact, train_examples],
                dtype=torch.float64,
                device=device,
            )
            if distributed and dist.is_initialized():
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            running_loss_total, sample_count_total, train_exact_total, train_examples_total = train_totals.tolist()
            avg_train_loss = running_loss_total / max(sample_count_total, 1)
            train_acc = train_exact_total / max(train_examples_total, 1)

            total_elapsed = time.time() - global_start
            total_steps = len(train_loader) * args.epochs
            steps_completed = min(previous_total_steps, total_steps)
            remaining_steps = total_steps - steps_completed
            avg_time_per_step_global = total_elapsed / max(steps_completed, 1)
            total_eta = remaining_steps * avg_time_per_step_global

            log_parts = [
                f"epoch={epoch}",
                f"train_loss={avg_train_loss:.4f}",
                f"train_acc={train_acc:.4f}",
                f"epoch_time={epoch_duration:.1f}s",
                f"eta_total={_format_eta(total_eta)}",
            ]

            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else args.learning_rate
            log_parts.append(f"lr={current_lr:.6f}")           
            if is_main_process:
                print(" | ".join(log_parts))

            if scheduler is not None:
                scheduler.step()

    finally:
        if distributed and dist.is_initialized():
            dist.barrier()

    if distributed and dist.is_initialized():
        dist.destroy_process_group()

    generate_predictions(
        model,
        eval_loader,
        device,
        img_size=args.image_size,
        attempt_nums=args.num_attempts,
        task_transform_resolver=get_eval_rot_transform_resolver(),
        fix_scale_factor=args.fix_scale_factor,
        disable_translation=args.disable_translation,
        if_fix_scale=args.disable_resolution_augmentation,
        save_name=args.eval_save_name + "_attempt_" + str(cur_attempt_idx),
        eval_split=args.eval_split,
        task_type=args.data_root.split("/")[-1],  # e.g., "ARC-AGI"
    )

def train(args: argparse.Namespace) -> None:
    distributed, rank, world_size, local_rank, device = init_distributed_mode(args)
    set_seed(args.seed + (rank if distributed else 0))

    train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler = build_dataloaders(
        args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    total_train_examples = len(train_dataset)

    if (not distributed) or rank == 0:
        print(f"Total training examples: {total_train_examples}")


    model_original = load_model_only(
        args=args, train_dataset=train_dataset, device=device, distributed=distributed, rank=rank, local_rank=local_rank
    )
    
    for attempt_idx in range(args.ttt_num_each):
        model = deepcopy(model_original)
        print(f"Starting test-time training attempt {attempt_idx + 1}/{args.ttt_num_each}...")
        ttt_once(model=model, device=device, distributed=distributed, rank=rank,
                train_loader=train_loader, train_sampler=train_sampler,
                eval_loader=eval_loader, cur_attempt_idx=attempt_idx)


if __name__ == "__main__":
    args = parse_args()

    # 本地 CPU 更稳
    args.no_compile = True
    args.no_amp = True
    args.distributed = False

    # 模型配置（与你训练 rvit 时一致）
    args.architecture = "rvit"
    args.depth = 10
    args.batch_size = 8
    args.image_size = 64
    args.patch_size = 2
    args.learning_rate = 1e-4
    args.weight_decay = 0
    args.embed_dim = 512
    args.num_heads = 8
    args.num_colors = 12
    args.lr_scheduler = "cosine"

    # 数据路径
    args.data_root = "raw_data/ARC-AGI"
    args.eval_save_name = "ARC_1_eval_ViT_nottt"
    args.num_attempts = 10

    # checkpoint
    args.resume_checkpoint = "saves/new_train_RARC/checkpoint_best.pt"

    # 如果你必须加载 task_token_embed 权重，就不要 skip
    # 如果仍报错，可以先改成 True 再排查
    args.resume_skip_task_token = False

    # 任务列表（可删减）
    file_names = [
        "af24b4cc","e1d2900e","903d1b4a","4e469f39","b1fc8b8e","2c737e39",
        "992798f6","00576224","48131b3c","60a26a3e","59341089","31d5ba1a",
        "e633a9e5","62ab2642","73c3b0d8","c663677b","c48954c1","08573cc6",
        "136b0064","929ab4e9","5b526a93","ef26cbf6","fafd9572","67c52801",
        "ad7e01d0","506d28a5","27a77e38","d492a647","72a961c9","fd4b2b02",
        "bf89d739","f5aa3634","b942fd60","d282b262","9772c176","ed74f2f2",
        "184a9768","94133066","256b0a75","e681b708","ce8d95cc","817e6c09",
        "7d18a6fb","1da012fc","310f3251","bf699163","917bccba","551d5bf1",
        "b457fec5","50a16a69","7953d61e","9a4bb226","c97c0139","d4b1c2b1",
        "1d398264","29700607","8597cfd7","a59b95c0","2f0c5170","bd14c3bf",
        "9caba7c3","e6de6e8f","da515329","31adaf00","f5c89df1","be03b35f",
        "833dafe3","6ea4a07e","2546ccf6","21f83797","696d4842","d94c3b52",
        "4ff4c9da","3979b1a8","bf32578f","d304284e","c62e2108","b0722778",
        "d19f7514","358ba94e","d017b73f","4c177718","b7999b51","e345f17b",
        "e4075551","50aad11f","66e6c45b","c074846d","0b17323b","4b6b68e5",
        "84db8fc4","ff72ca3e","8ee62060","52fd389e","ae58858e","fea12743",
        "0f63c0b9","e99362f0","195ba7dc","f3cdc58f","a8610ef7","e760a62e",
        "aa300dc3","ea9794b1","e41c6fd3","5d2a5c43","e66aafb8","ca8de6ea",
        "19bb5feb","7c8af763","e872b94a","6f473927","ac605cbb","ac3e2b04",
        "0e671a1a","ac0c5833","fb791726","351d6448","ce039d91","45bbe264",
        "332efdb3","c64f1187","5b6cbef5","1d0a4b61","42918530","7bb29440",
        "3a301edc","896d5239","505fff84","cfb2ce5a","140c817e","69889d6e",
        "20818e16","9b2a60aa","626c0bcc","a57f2f04","477d2879","05a7bcf2",
        "81c0276b","ba9d41b8","e133d23d","604001fa","3ee1011a","85b81ff1",
        "17b80ad2","9b365c51","e7dd8335","2a5f8217","712bf12e","84f2aca1",
        "ac2e8ecf","e2092e0c","33b52de3","5833af48","319f2597","aa18de87",
        "cb227835","e74e1818","15663ba9","b4a43f3b","281123b4","fc754716",
        "e5790162","94414823","642d658d","96a8c0cd","2697da3f","e9b4f6fc",
        "bcb3040b","55783887","1acc24af","981571dc","705a3229","1c02dbbe",
        "ca8f78db","1e97544e","92e50de0","e57337a4","4852f2fa","7d1f7ee8",
        "e1baa8a4","14754a24","62b74c02","7d419a02","94be5b80","68b67ca3",
        "2072aba6","fe9372f3","137f0df0","c6e1b8da","16b78196","1c0d0a4b",
        "f0afb749","de493100","1990f7a8","423a55dc","2753e76c","f21745ec",
        "bc4146bd","79fb03f4","3d31c5b3","c35c1b4c","cf133acc","da2b0fe3",
        "15696249","0c9aba6e","e7a25a18","d5c634a2","414297c0","009d5c81",
        "0becf7df","f3e62deb","58743b76","9b4c17c4","891232d6","4e45f183",
        "c7d4e6ad","a04b2602","d37a1ef5","25094a63","0c786b71","f3b10344",
        "b7f8a4d8","b7cb93ac","b15fca0b","1a6449f1","67636eac","f823c43c",
        "27f8ce4f","fd096ab6","0a1d4ef5","6a11f6da","8fbca751","6ad5bdfd",
        "3ed85e70","09c534e7","642248e4","9f27f097","50f325b5","88207623",
        "45737921","11e1fe23","aee291af","90347967","e7b06bea","03560426",
        "e7639916","e21a174a","4aab4007","c658a4bd","5783df64","1c56ad9f",
        "c1990cce","93c31fbe","5af49b42","7e02026e","2685904e","c8b7cc0f",
        "15113be4","b20f7c8b","575b1a71","e0fb7511","3b4c2228","32e9702f",
        "ccd554ac","af22c60d","bbb1b8b6","f9d67f8b","0d87d2a6","ed98d772",
        "9ddd00f0","070dd51e","9356391f","4acc7107","47996f11","8dae5dfc",
        "e5c44e8f","e9bb6954","dc2aa30b","d2acf2cb","292dd178","f9a67cb5",
        "20981f0e","12997ef3","103eff5b","770cc55f","0692e18c","8719f442",
        "e88171ec","95a58926","639f5a19","40f6cd08","3f23242b","d47aa2ff",
        "5b692c0f","a934301b","a3f84088","72207abc","73182012","0a2355a6",
        "a680ac02","58e15b12","64a7c07e","e9c9d9a1","12eac192","dc2e9a9d",
        "ecaa0ec1","4cd1b7b2","7c9b52a0","3391f8c0","9c56f360","0607ce86",
        "97239e3d","a406ac07","baf41dbf","c3202e5a","13713586","42a15761",
        "e619ca6e","e69241bd","d56f2372","5207a7b5","8ba14f53","b0f4d537",
        "aa4ec2a5","3490cc26","9def23fe","d4c90558","12422b43","99306f82",
        "516b51b7","cad67732","f4081712","212895b5","67b4a34d","845d6e51",
        "66f2d22f","73ccf9c2","f83cb3f6","d931c21c","17cae0c1","22a4bbc2",
        "0934a4d8","8cb8642d","1e81d6f9","85fa5666","0bb8deee","55059096",
        "4364c1c4","4f537728","aab50785","9110e3c5","762cd429","7039b2d7",
        "bb52a14b","ea959feb","dd2401ed","48f8583b","2b01abd0","b7fb29bc",
        "8b28cd80","782b5218","5a5a2103","759f3fd3","e9ac8c9e","9c1e755f",
        "37d3e8b2","9bebae7a","1a2e2828","34b99a2b","60c09cac","f45f5ca7",
        "7ee1c6ea","54db823b","00dbd492","b9630600","2037f2c7","c87289bb",
        "692cd3b6","79369cc6","c92b942c","6df30ad6","a096bf4d","e78887d1",
        "5ffb2104","5289ad53","8a371977","f0df5ff0","df8cc377","cd3c21df",
        "963f59bc","3194b014","93b4f4b3","2c0b0aff","456873bc","8e2edd66",
        "f8be4b64","e95e3d8e","695367ec","18419cfa"
    ]

    # for file_name in file_names:
    #     print(f"Processing {file_name} ...")
    #     args.train_split = f"eval_color_permute_ttt_9/{file_name}"
    #     args.eval_split = f"eval_color_permute_ttt_9/{file_name}"
    #     train(args)



    distributed, rank, world_size, local_rank, device = init_distributed_mode(args)
    train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler = build_dataloaders(
        args, distributed=distributed, rank=rank, world_size=world_size
    )

    model = load_model_only(
        args=args,
        train_dataset=train_dataset,
        device=device,
        distributed=distributed,
        rank=rank,
        local_rank=local_rank
    )

    # 直接推理（不训练）
    generate_predictions(
        model,
        eval_loader,
        device,
        img_size=args.image_size,
        attempt_nums=args.num_attempts,
        task_transform_resolver=get_eval_rot_transform_resolver(),
        fix_scale_factor=args.fix_scale_factor,
        disable_translation=args.disable_translation,
        if_fix_scale=args.disable_resolution_augmentation,
        save_name=args.eval_save_name,
        eval_split=args.eval_split,
        task_type=args.data_root.split("/")[-1],
    )