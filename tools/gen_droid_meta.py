import argparse
import glob
import mmengine
import h5py
import io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm


def _open_h5(path: str) -> h5py.File:
    """Open HDF5 from local FS or remote backend via mmengine.fileio."""
    try:
        return h5py.File(path, "r")
    except OSError:
        from mmengine import fileio
        return h5py.File(io.BytesIO(fileio.get(path)), "r")


def read_instruction_from_h5(filepath: str, instruction_key: str = "language_instruction") -> str:
    """Read instruction from HDF5 file."""
    try:
        with _open_h5(filepath) as f:
            if instruction_key in f.attrs:
                v = f.attrs[instruction_key]
                lang = v.decode() if isinstance(v, bytes) else v
            else:
                ds = f[instruction_key]
                v = ds[()]
                lang = v.decode() if getattr(ds, "shape", ()) == () else v[0].decode()
            
            if isinstance(lang, list):
                lang = lang[0] if lang else ""
            if isinstance(lang, bytes):
                lang = lang.decode("utf-8")
            
            return lang.strip()
    except Exception as e:
        print(f"Warning: Could not read instruction from {filepath}: {e}")
        return ""


def check_file_has_instruction(filepath: str, instruction_key: str = "language_instruction") -> tuple:
    """Check if a file has a non-empty instruction. Returns (filepath, has_instruction)."""
    instruction = read_instruction_from_h5(filepath, instruction_key)
    return (filepath, bool(instruction))


def generate_droid_meta(data_dir: str, output_path: str, num_workers: int = 32, filter_empty: bool = True):
    """Generate metadata for Droid dataset with parallel processing.
    
    Args:
        data_dir: Directory containing HDF5 files
        output_path: Path to save metadata JSON
        num_workers: Number of parallel workers for processing files
        filter_empty: Whether to filter out files with empty instructions
    """
    meta = dict()
    meta["observation_key"] = ["observation/exterior_image_1_left", "observation/wrist_image_left"]
    meta["dataset_name"] = "Droid-Left"
    meta["language_instruction_key"] = "language_instruction"
    
    filelist = glob.glob(f"{data_dir}/episode_*.hdf5")
    filelist = sorted(filelist)
    
    print(f"Total files found: {len(filelist)}")
    
    if filter_empty and filelist:
        print(f"Filtering files with empty instructions using {num_workers} workers...")
        
        filtered_filelist = []
        empty_count = 0
        
        check_func = partial(check_file_has_instruction, instruction_key=meta["language_instruction_key"])
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(check_func, filepath): filepath for filepath in filelist}
            
            for future in tqdm(as_completed(futures), total=len(filelist), desc="Processing files"):
                try:
                    filepath, has_instruction = future.result()
                    if has_instruction:
                        filtered_filelist.append(filepath)
                    else:
                        empty_count += 1
                except Exception as e:
                    print(f"\nError processing file: {e}")
                    empty_count += 1
        
        filtered_filelist = sorted(filtered_filelist)
        meta['datalist'] = filtered_filelist
        
        print(f"Files with empty instructions: {empty_count}")
        print(f"Files with valid instructions: {len(filtered_filelist)}")
    else:
        meta['datalist'] = filelist
        print(f"Using all files without filtering")
    
    print(f"Sample files: {meta['datalist'][:5]}")
    print(f"Total files in metadata: {len(meta['datalist'])}")
    
    mmengine.dump(meta, output_path)
    print(f"Metadata saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate metadata file for Droid dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str((Path(__file__).resolve().parent.parent / "data" / "openX" / "droid_hdf5")),
        help="Directory containing Droid HDF5 files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str((Path(__file__).resolve().parent.parent / "data" / "droid_meta.json")),
        help="Path to save metadata JSON"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel workers for processing files (default: 32)"
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Do not filter files with empty instructions"
    )
    
    args = parser.parse_args()
    
    generate_droid_meta(
        data_dir=args.data_dir,
        output_path=args.output_path,
        num_workers=args.num_workers,
        filter_empty=not args.no_filter
    )


if __name__ == "__main__":
    main()
