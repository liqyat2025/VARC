from utils.data_augmentation import augment_raw_data_split_per_task

if __name__ == "__main__":
    augment_raw_data_split_per_task(
        dataset_root="raw_data/ARC-AGI",
        split="evaluation",
        output_subdir="eval_color_permute_ttt_9", #输出子目录名称
        num_permuate=9, #每个增强任务生成的颜色排列数量
        only_basic=True, #只应用基础增强（旋转、翻转）
    )

    augment_raw_data_split_per_task(
        dataset_root="raw_data/ARC-AGI-2",
        split="evaluation",
        output_subdir="eval_color_permute_ttt_9",
        num_permuate=9,
        only_basic=True,
    )