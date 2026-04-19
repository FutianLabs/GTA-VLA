import base64
import json
import os
import pandas as pd
from tqdm import tqdm
import subprocess
import uuid
import h5py
import numpy as np
import imageio
import io
import multiprocessing
from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from multiprocessing import Manager


class VideoDataTSVWriter:
    """
    Basic class for organizing video data into TSV format.
    Inputs:
        tsv_dir: string, the directory to save the tsv file.
        tsv_name: string, the name of the tsv file. After processing, there will be 
            tsv_name.images.tsv: recording the video data.
            tsv_name.annotations.tsv: recording the annotation data.
            tsv_name.images.tsv.lineidx: recording the line index of the video data.
            tsv_name.annotations.tsv.lineidx: recording the line index of the annotation data.
        video_encoding: string, the encoding format of the video. Default is "h264".
    """
    def __init__(self, tsv_dir: str, tsv_name: str, video_encoding="h264") -> None:
        os.makedirs(tsv_dir, exist_ok=True)
        tsv_basename = os.path.join(tsv_dir, tsv_name)
        self.tsv_name = tsv_name
        self.tsv_basename = tsv_basename
        self.tsv_video_path       = tsv_basename + '.images.tsv'
        self.video_lineidx_path   = tsv_basename + '.images.tsv.lineidx'
        self.tsv_anno_path       = tsv_basename + '.annotations.tsv'
        self.anno_lineidx_path   = tsv_basename + '.annotations.tsv.lineidx'
        self._open_file()
        self.video_encoding = video_encoding

    def _open_file(self):
        mode = "w+"
        self._tsv_video      = open(self.tsv_video_path, mode, encoding="utf-8")
        self._video_lineidx  = open(self.video_lineidx_path, mode, encoding="utf-8")
        self._tsv_anno      = open(self.tsv_anno_path, mode, encoding="utf-8")
        self._anno_lineidx  = open(self.anno_lineidx_path, mode, encoding="utf-8")

    def _close_file(self):
        self._tsv_video.close()
        self._video_lineidx.close()
        self._tsv_anno.close()
        self._anno_lineidx.close()    

    def encode_video(self, video_dict_sample) -> str:
        """
        1. Transform the video to the specified encoding format.
        2. Then encode the videos to base64 format. And return a dict, recording different video sources.

        Input:
            video_dict_sample: {
                "camera_name_1": video_path_1,
                "camera_name_2": video_path_2,
                ...
            }
        Output:
            video_base64_dict: {
                "camera_name_1": base64(video_encoding(video_1)),
                "camera_name_2": base64(video_encoding(video_2)),
                ...
            }
        """
        def encode_video_to_h264_in_memory(path_video):
            """
            Judge if the video is encoded as H.264.
            If yes, return the original path.
            If not, encode it to H.264 and save it to /dev/shm, return the temp path.
            The caller should delete the temp file after use.
            """
            # 1. 检查编码
            try:
                cmd = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    path_video
                ]
                codec = subprocess.check_output(cmd, text=True).strip()
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ffprobe failed: {e}")

            is_tmp_file = False

            if codec.lower() == "h264":
                return path_video, is_tmp_file  # 已经是 H.264

            # 2. 生成 /dev/shm 临时文件路径
            tmp_file_path = f"/dev/shm/{uuid.uuid4().hex}.mp4"  # TODO: make sure the server has this directory.

            try:
                subprocess.run(["ffmpeg", "-i", path_video, "-c:v", "h264", tmp_file_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                raise RuntimeError(f"ffmpeg encoding failed: {e.stderr.decode()}")
            is_tmp_file = True
            return tmp_file_path, is_tmp_file
        
        def get_video_base64(video):
            if isinstance(video, str):
                with open(video, "rb") as f:
                    video_bytes = f.read()
            elif isinstance(video, bytes):
                video_bytes = video
            else:
                raise NotImplementedError
            video_base64 = base64.b64encode(video_bytes).decode("utf-8")
            return video_base64
        
        video_base64_dict = {}
        for video_name, video in video_dict_sample.items():
            if isinstance(video, str):
                raise NotImplementedError
                if self.video_encoding == "h264":
                    encoded_video_path, is_tmp_file = encode_video_to_h264_in_memory(video_path)
                else:
                    raise NotImplementedError
                video_base64_dict[video_name] = get_video_base64(encoded_video_path)
                if is_tmp_file:
                    os.remove(encoded_video_path)
            elif isinstance(video, bytes):
                video_base64_dict[video_name] = get_video_base64(video)
        return video_base64_dict
        
    def encode_annotation(self, annotation_dict: dict) -> str:
        """
        Origanize the annotation into a dict of string / list.
        By default, the annotation is a dict, recording annotations with type as string or list, so no action required.
        """
        return annotation_dict
    
    def write_video(self, data_id, encoded_video_dict):
        """
        Dump the video_dict into string, and write the videos to the TSV file as one line.
        Input:
            data_id: the id of the sample.
            encoded_video_dict: a dict, recording different video sources.
                {
                    "camera_name_1": base64(video_encoding(video_1)),
                    "camera_name_2": base64(video_encoding(video_2)),
                    ...
                }
        Output:
            video_offset: the offset of the video's (line) start position in the TSV file.
        """

        video_offset = self._tsv_video.tell()
        self._tsv_video.write("\t".join([str(data_id), json.dumps(encoded_video_dict)])+"\n")
        # self._tsv_video.flush()
        # os.fsync(self._tsv_video.fileno()) 
        self._video_lineidx.write(str(video_offset)+"\n")
        return video_offset
    
    def write_annotation(self, video_offset, encoded_annotation):
        """
        Dump the annotation dict into string, and write the annotation to the TSV file.
        Input:
            video_offset: the offset of the annotation's corresponding video's start position in the TSV file.
            encoded_annotation: a dict, recording different annotation sources.
        Output:
            anno_offset: the offset of the annotation's (line) start position in the TSV file.
        """
        anno_offset  = self._tsv_anno.tell()
        self._tsv_anno.write("\t".join([str(video_offset), json.dumps(encoded_annotation)])+"\n")
        # self._tsv_anno.flush()
        # os.fsync(self._tsv_anno.fileno()) 
        self._anno_lineidx.write(str(anno_offset)+"\n")
        return anno_offset
    
    def write_tsv(self):
        """
        The enterpoint, write the video and annotation data to the TSV file.
        """
        print(f"User should implement their own write_tsv method.")
        raise NotImplementedError


class BridgeLerobotTSVWriter(VideoDataTSVWriter):
    """
    This class is used to organize the video / annotation to the TSV file for the Bridge Lerobot dataset.
    """
    def __init__(self, ori_dataset_dir: str, tsv_dir: str, tsv_name: str, video_encoding="h264") -> None:
        self.ori_dataset_dir = ori_dataset_dir
        video_dict = self.get_video_dict()
        annotation_dict = self.get_annotation_dict()
        super().__init__(video_dict, annotation_dict, tsv_dir, tsv_name, video_encoding)
    
    def get_video_dict(self) -> list:
        """
        Get the video dict for each data sample. Formated as dict of dict. (To align with the annotation dict.)
        {
            "sample_name": {
                "camera_name_1": video_path_1,
                "camera_name_2": video_path_2,
                ...
            },
            "sample_name_2": {
                "camera_name_1": video_path_1,
                "camera_name_2": video_path_2,
                ...
            },
            ...
        }
        """
        ori_video_dir = os.path.join(self.ori_dataset_dir, "videos")
        chuncks = os.listdir(ori_video_dir)
        video_dict = {}
        for chunck in chuncks:
            dir_chunck = os.path.join(ori_video_dir, chunck)
            views = os.listdir(dir_chunck)
            episodes = set([])
            for view in views:
                dir_view = os.path.join(dir_chunck, view)
                videos = os.listdir(dir_view)
                episodes = episodes | set(videos)
            for episode in episodes:
                episode_video_dict = {}
                for view_name in views:
                    path_view = os.path.join(dir_chunck, view_name, episode)
                    if os.path.exists(path_view):
                        episode_video_dict[view_name] = path_view
                episode_name = f"{chunck}_{episode.split('.')[0]}"
                video_dict[episode_name] = episode_video_dict
        return video_dict

    def get_annotation_dict(self) -> list:
        """
        Get the annotation dict for each data sample. Formated as dict of dict. (To align with the video dict.)
        {
            "sample_name_1": {
                "anno_item_1": annotation_1,
                "anno_item_2": annotation_2,
                ...
            },
            "sample_name_2": {
                "anno_item_1": annotation_1,
                "anno_item_2": annotation_2,
                ...
            },
            ...
        }
        The annotations are aggregated along the temporal dimension. Eg, the shape of action = [T, D], where D is the dimension of action.

        To simplify writing of tsv file, the annotations are formated as list or string, not np.array or torch.Tensor.
        """
        def load_parquet_data(path_parquet):
            """
            load .parquet data, and return a dict.
            """
            data = pd.read_parquet(path_parquet)
            return data.to_dict(orient="records")
        
        def aggregate_timedimension(data: dict) -> dict:
            """
            aggregate the data by the time dimension.
            """
            states = [time_sample["observation.state"].tolist() for time_sample in data]
            actions = [time_sample["action"].tolist() for time_sample in data]
            time_stamps = [time_sample["timestamp"] for time_sample in data]
            frame_indexes = [time_sample["frame_index"] for time_sample in data]
            episode_indexes = [time_sample["episode_index"] for time_sample in data]
            indexes = [time_sample["index"] for time_sample in data]
            task_indexes = [time_sample["task_index"] for time_sample in data]
            return {
                "observation.state": states,
                "action": actions,
                "timestamp": time_stamps,
                "frame_index": frame_indexes,
                "episode_index": episode_indexes,
                "index": indexes,
                "task_index": task_indexes,
            }

        ori_annotation_dir = os.path.join(self.ori_dataset_dir, "data")
        chuncks = os.listdir(ori_annotation_dir)
        annotation_dict = {}
        for chunck in chuncks:
            dir_chunck = os.path.join(ori_annotation_dir, chunck)
            episodes = os.listdir(dir_chunck)
            for episode in episodes:
                path_episode = os.path.join(dir_chunck, episode)
                episode_anno = load_parquet_data(path_episode)
                episode_anno = aggregate_timedimension(episode_anno)
                episode_name = f"{chunck}_{episode.split('.')[0]}"
                episode_anno["episode_name"] = episode_name
                annotation_dict[episode_name] = episode_anno
        return annotation_dict


class HDF5TSVWriter(VideoDataTSVWriter):
    def __init__(self, 
        ori_dataset_dir: str, 
        tsv_dir: str, tsv_name: str, 
        video_encoding="h264",
        annotation_required_keys=[],
        language_instruction_key="instruction",
        parallel_num=None, parallel_id=None,
        debug=False,
        progress_callback=None,
        filter_empty_instruction=True,  # 新增：是否过滤空指令
    ) -> None:
        self.ori_dataset_dir = ori_dataset_dir
        self.annotation_required_keys = annotation_required_keys
        self.parallel_num = parallel_num
        self.language_instruction_key = language_instruction_key
        self.parallel_id = parallel_id
        self.debug = debug
        self.progress_callback = progress_callback
        self.filter_empty_instruction = filter_empty_instruction
        super().__init__(tsv_dir, tsv_name, video_encoding)
    
    def encode_video_to_bytes(self, frames: np.ndarray, fps=30, codec="libx264"):
        """
        Encode a video (T, H, W, 3) to H.264 bytes using FFmpeg subprocess.
        """
        import tempfile
        
        height, width = frames.shape[1], frames.shape[2]
        
        # 使用 ffmpeg 子进程进行编码
        # 输入：原始 RGB 帧数据通过 stdin
        # 输出：H.264 编码的 MP4 通过 stdout
        cmd = [
            '/root/miniconda3/bin/ffmpeg',  # 使用系统FFmpeg（支持libx264）
            '-y',  # 覆盖输出文件
            '-f', 'rawvideo',  # 输入格式：原始视频
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',  # 分辨率
            '-pix_fmt', 'rgb24',  # 输入像素格式
            '-r', str(fps),  # 帧率
            '-i', '-',  # 从 stdin 读取
            '-c:v', 'libx264',  # 使用 H.264 编码
            '-pix_fmt', 'yuv420p',  # 输出像素格式（兼容性好）
            '-crf', '10',  # 质量控制（0-51，越小质量越好）- 极高质量，接近完美
            '-preset', 'medium',  # 编码速度预设 - 平衡速度和质量
            '-movflags', 'frag_keyframe+empty_moov',  # 支持流式输出到 stdout
            '-f', 'mp4',  # 输出格式
            '-'  # 输出到 stdout
        ]
        
        # 将帧数据转换为字节
        raw_frames = frames.astype(np.uint8).tobytes()
        
        # 运行 ffmpeg
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        video_bytes, stderr = process.communicate(input=raw_frames)
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed: {stderr.decode()}")
        
        return video_bytes
    
    def read_instruction(self, f: h5py.File) -> str:
        raw_key = self.language_instruction_key
        keys = raw_key if isinstance(raw_key, list) else [raw_key]
        for key in keys:
            if key in f.attrs:
                v = f.attrs[key]
                result = v.decode() if isinstance(v, bytes) else v
                if isinstance(result, str) and result.strip():
                    return result
            elif key in f:
                ds = f[key]
                v = ds[()]
                result = v.decode() if getattr(ds, "shape", ()) == () else v[0].decode()
                if isinstance(result, str) and result.strip():
                    return result
        return ""

    def get_data_onesample(self, path_hdf5):
        observations = {}
        annotations = {}
        with h5py.File(path_hdf5, "r") as f:
            # get video
            for camera_name in list(list(f["observation"].keys())):
                video = f["observation"][camera_name]
                video_bytes = self.encode_video_to_bytes(np.array(video))
                observations[camera_name] = video_bytes
            # get annotation
            for anno_key in self.annotation_required_keys:
                # 保持原始数据类型，使用 float32 以节省空间并保持精度
                data = np.array(f[anno_key])
                # 如果是 float64，转换为 float32
                if data.dtype == np.float64:
                    data = data.astype(np.float32)
                annotations[anno_key] = data.tolist()
            annotations["sample_name"] = path_hdf5
            annotations["episode_name"] = path_hdf5.split("/")[-1].split(".")[0]
            annotations["language_instruction"] = self.read_instruction(f)
        return observations, annotations
        
    def write_tsv_onesample(self, sample_id, observations, annotations):
        encoded_video = self.encode_video(observations)
        encoded_annotation = self.encode_annotation(annotations)
        video_offset = self.write_video(sample_id, encoded_video)
        anno_offset = self.write_annotation(video_offset, encoded_annotation)

    def get_ori_file_list(self):
        file_list = sorted(os.listdir(self.ori_dataset_dir))
        if self.parallel_num is not None:
            # 连续分块：每个进程处理一个连续的文件范围
            total_files = len(file_list)
            chunk_size = total_files // self.parallel_num
            remainder = total_files % self.parallel_num
            
            # 均匀分配：前 remainder 个进程每个多处理一个文件
            if self.parallel_id < remainder:
                start_idx = self.parallel_id * (chunk_size + 1)
                end_idx = start_idx + chunk_size + 1
            else:
                start_idx = self.parallel_id * chunk_size + remainder
                end_idx = start_idx + chunk_size
            
            file_list = file_list[start_idx:end_idx]
        return file_list

    def write_tsv(self):
        hdf5_files = self.get_ori_file_list()
        if self.debug:
            hdf5_files = hdf5_files[:10]
        
        total_files = len(hdf5_files)
        
        # 如果有进度回调函数，先初始化总数
        if self.progress_callback:
            self.progress_callback(0, total_files)
        
        if self.parallel_num is None:
            tqdm_desc = f"Writing TSV for {self.tsv_name} [{total_files} Samples]" 
        else:
            tqdm_desc = f"Writing TSV for {self.tsv_name} (Parallel {self.parallel_id}/{self.parallel_num}) [{total_files}]"
        
        # 如果没有进度回调，使用tqdm显示进度
        iterable = enumerate(hdf5_files)
        if not self.progress_callback:
            iterable = tqdm(iterable, desc=tqdm_desc, total=total_files)
        
        # 统计信息
        skipped_empty_instruction = 0
        written_samples = 0
        
        for file_idx, hdf5_file in iterable:
            try:
                path_hdf5 = os.path.join(self.ori_dataset_dir, hdf5_file)
                observations, annotations = self.get_data_onesample(path_hdf5)
                
                # 检查是否需要过滤空指令
                if self.filter_empty_instruction:
                    instruction = annotations.get("language_instruction", "")
                    # 检查指令是否为空
                    if not instruction or (isinstance(instruction, str) and instruction.strip() == ""):
                        skipped_empty_instruction += 1
                        if not self.progress_callback:
                            # 更新tqdm描述显示跳过的数量
                            if hasattr(iterable, 'set_postfix'):
                                iterable.set_postfix({
                                    'written': written_samples,
                                    'skipped': skipped_empty_instruction
                                })
                        continue
                
                # 写入TSV，使用written_samples作为sample_id以保持连续性
                self.write_tsv_onesample(written_samples, observations, annotations)
                written_samples += 1
                
                # 更新进度回调
                if self.progress_callback:
                    self.progress_callback(file_idx + 1, total_files)
                    
            except Exception as e:
                print(f"Error processing file {hdf5_file}: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        self._close_file()
        
        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"TSV Writing Summary:")
        print(f"  Total files processed: {total_files}")
        print(f"  Written samples: {written_samples}")
        if self.filter_empty_instruction:
            print(f"  Skipped (empty instruction): {skipped_empty_instruction}")
            print(f"  Keep rate: {written_samples/total_files*100:.2f}%")
        print(f"{'='*60}")
    


class MultiProcessorTSVWriterWrapper:
    def __init__(self, parallel_num=None, tsv_writer=None, **kwargs) -> None:
        if not parallel_num or parallel_num < 1:
            raise ValueError(f"parallel_num must be >= 1, got {parallel_num}")
        if not tsv_writer:
            raise ValueError("tsv_writer cannot be None")
        
        self.parallel_num = parallel_num
        self.tsv_writer_class = tsv_writer
        self.kwargs = kwargs
    
    @staticmethod
    def _write_single_worker(tsv_writer_class, parallel_num, parallel_id, kwargs, progress_dict=None):
        """子进程工作函数，支持进度报告"""
        kwargs_ = deepcopy(kwargs)
        kwargs_["tsv_name"] = kwargs_["tsv_name"] + f"_parallel_{parallel_num}_{parallel_id}"
        kwargs_["parallel_num"] = parallel_num
        kwargs_["parallel_id"] = parallel_id
        
        # 为worker添加进度回调函数
        if progress_dict is not None:
            kwargs_["progress_callback"] = lambda current, total: progress_dict.update({
                parallel_id: {"current": current, "total": total}
            })
        
        writer = tsv_writer_class(**kwargs_)
        try:
            writer.write_tsv()
            # 返回统计信息（如果有的话）
            stats = {
                'success': True,
                'parallel_id': parallel_id,
            }
            return parallel_id, stats
        except Exception as e:
            return parallel_id, {'success': False, 'error': str(e)}
        finally:
            if hasattr(writer, '_close_file'):
                writer._close_file()
    
    def write_tsv(self, max_workers=None):
        """并行写入TSV文件，带进度条"""
        if self.parallel_num == 1:
            self._write_single_worker(self.tsv_writer_class, 1, 0, self.kwargs)
            return
        
        max_workers = min(max_workers or self.parallel_num, self.parallel_num)
        
        # 创建进程间共享的进度字典
        manager = Manager()
        progress_dict = manager.dict()
        
        # 初始化每个worker的进度
        for i in range(self.parallel_num):
            progress_dict[i] = {"current": 0, "total": 1}
        
        # 创建每个worker的进度条
        pbars = {}
        for i in range(self.parallel_num):
            pbars[i] = tqdm(
                total=1,
                desc=f"Worker {i:2d}",
                position=i,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._write_single_worker,
                    self.tsv_writer_class,
                    self.parallel_num,
                    i,
                    self.kwargs,
                    progress_dict
                ): i for i in range(self.parallel_num)
            }
            
            # 启动进度监控
            failed = []
            completed_workers = set()
            
            while len(completed_workers) < self.parallel_num:
                # 更新所有进度条
                for worker_id, pbar in pbars.items():
                    if worker_id in progress_dict:
                        prog_data = progress_dict[worker_id]
                        current = prog_data["current"]
                        total = prog_data["total"]
                        
                        # 更新进度条的total和当前值
                        if pbar.total != total:
                            pbar.reset(total=total)
                        pbar.n = current
                        pbar.refresh()
                
                # 检查已完成的任务
                for future in list(futures.keys()):
                    if future.done() and futures[future] not in completed_workers:
                        worker_id = futures[future]
                        completed_workers.add(worker_id)
                        parallel_id, result = future.result()
                        
                        # 检查结果
                        if isinstance(result, dict):
                            if not result.get('success', False):
                                failed.append((parallel_id, result.get('error', 'Unknown error')))
                        elif result is not True:
                            failed.append((parallel_id, result))
                        
                        # 确保进度条显示为完成状态
                        if worker_id in pbars:
                            pbars[worker_id].n = pbars[worker_id].total
                            pbars[worker_id].refresh()
                
                time.sleep(0.1)  # 避免过于频繁的更新
            
            # 关闭所有进度条
            for pbar in pbars.values():
                pbar.close()
            
            if failed:
                error_msg = "\n".join(f"Process {pid}: {err}" for pid, err in failed)
                raise RuntimeError(f"Failed processes:\n{error_msg}")


if __name__ == "__main__":
    
    # TODO: judges the min number of multi-processor

    # dir_dataset = "/comp_robot/lihongyang/code/ProtectedCode/VideoDataAnnotation/VideoTSVConstruction_NoSample/test_data/VLA_DATA/bridge_orig_lerobot"
    # dir_tsv     = "/comp_robot/lihongyang/code/ProtectedCode/VideoDataAnnotation/VideoTSVConstruction_NoSample/test_data/VLA_DATA/bridge_orig_lerobot_tsv"
    # data_name   = "BridgeLerobot"
    # video_encoding = "h264"
    # tsvwriter = BridgeLerobotTSVWriter(dir_dataset, dir_tsv, data_name, video_encoding)

    dir_dataset = "/VLA-Data/scripts/lianqing/data/openX/x-vla/bridge"
    dir_tsv_base = "/VLA-Data/scripts/lianqing/data/openX/x-vla/bridge_tsv_lianqing_lang"
    data_name   = "BridgeHDF5"
    video_encoding = "h264"
    DEBUGMODE = False # if True, only write the first 10 samples
    FILTER_EMPTY_INSTRUCTION = False  # Set to True to filter empty instructions
    
    # Automatically add '_filter' suffix to dir if filtering is enabled
    if FILTER_EMPTY_INSTRUCTION:
        dir_tsv = dir_tsv_base + "_filter"
    else:
        dir_tsv = dir_tsv_base
    
    print(f"Configuration:")
    print(f"  Source: {dir_dataset}")
    print(f"  Target: {dir_tsv}")
    print(f"  Filter empty instructions: {FILTER_EMPTY_INSTRUCTION}")
    print(f"  Debug mode: {DEBUGMODE}")
    print()
    
    # annotation_required_keys = ['abs_action_6d', 'actions', 'dones', 'rewards']  # libero
    annotation_required_keys = ['action', 'proprio']  # bridge
    # tsvwriter = LiberoHDF5TSVWriter(
    #     dir_dataset, 
    #     dir_tsv, data_name, 
    #     video_encoding, 
    #     annotation_required_keys=annotation_required_keys
    # )
    tsvwriter = MultiProcessorTSVWriterWrapper(
        parallel_num=40,
        tsv_writer=HDF5TSVWriter,
        ori_dataset_dir=dir_dataset,
        tsv_dir=dir_tsv,
        tsv_name=data_name,
        video_encoding=video_encoding,
        annotation_required_keys=annotation_required_keys,
        debug=DEBUGMODE,
        filter_empty_instruction=FILTER_EMPTY_INSTRUCTION,
    )
    
    tsvwriter.write_tsv()