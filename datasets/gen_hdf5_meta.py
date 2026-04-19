import os
import json
import argparse
import importlib.util
from typing import Dict, List

def load_domain_config(domain_config_path: str):
    spec = importlib.util.spec_from_file_location("domain_config", domain_config_path)
    domain_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(domain_config)
    return domain_config

def collect_episodes(root_dir: str) -> Dict[str, List[str]]:
    """
    遍历 root_dir 下所有本体，收集每个本体下的所有 episode 路径
    返回: {本体名: [episode路径, ...], ...}
    """
    result = {}
    for subdir in os.listdir(root_dir):
        abs_subdir = os.path.join(root_dir, subdir)
        if not os.path.isdir(abs_subdir):
            continue
        # 只处理 h5_ 开头的本体目录
        if not subdir.startswith("h5_"):
            continue
        episode_list = []
        for dirpath, dirnames, filenames in os.walk(abs_subdir):
            # 只收集最底层的目录（假设每个 episode 是一个目录）
            if not dirnames:  # 没有子目录
                episode_list.append(dirpath)
        if episode_list:
            result[subdir] = episode_list
    return result

def main(args):
    # 可视化所有本体文件夹名
    all_folders = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    print("Found folders:", all_folders)

    # 本体归类映射
    body_map = {
        "robomind-franka": lambda name: name.startswith("h5_franka") and "dual" not in name,
        "robomind-franka-dual": lambda name: name.startswith("h5_franka") and "dual" in name,
        "robomind-ur": lambda name: "ur" in name,
        "robomind-agilex": lambda name: "agilex" in name,
    }

    for config_name, match_func in body_map.items():
        # 找到所有属于该本体的视角文件夹
        matched_folders = [f for f in all_folders if match_func(f)]
        if not matched_folders:
            print(f"No folders found for {config_name}")
            continue
        episode_list = []
        for folder_name in matched_folders:
            abs_folder = os.path.join(args.data_root, folder_name)
            for dirpath, dirnames, filenames in os.walk(abs_folder):                                                                
                if not dirnames:
                    episode_list.append(dirpath)
        if not episode_list:
            print(f"No episodes found for {config_name}")
            continue
        meta = {
            "dataset_name": config_name,
            "domain_folders": matched_folders,
            "datalist": episode_list,
            "observation_key": args.observation_key,
            "language_instruction_key": args.language_instruction_key
        }
        out_path = os.path.join(args.output_dir, f"{config_name}_meta.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Meta file saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/VLA-Data/scripts/lingyiran/data/x-humanoid-robomind/RoboMIND",type=str, required=True, help="RoboMIND数据根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="meta文件输出文件夹")
    parser.add_argument("--observation_key", default="observations/rgb_images/camera_top/rgb", type=str, nargs="+", help="观测键")
    parser.add_argument("--language_instruction_key", type=str, default="language_instruction", help="语言指令键")
    args = parser.parse_args()
    main(args)