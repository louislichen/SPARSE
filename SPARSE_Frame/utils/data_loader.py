from datasets import load_dataset
from tqdm import tqdm

def get_calib_data(data_path, tokenizer, num_samples, seq_len):
    print(f"[Data] Loading from {data_path}...")
    try:
        dataset = load_dataset("json", data_files={"train": data_path}, split="train", streaming=True)
        iter_ds = iter(dataset)
        samples = []
        pbar = tqdm(total=num_samples, desc="Collecting Samples")
        while len(samples) < num_samples:
            try:
                item = next(iter_ds)
                text = item.get('text', '')
                if len(text) > seq_len:
                    samples.append(text)
                    pbar.update(1)
            except StopIteration:
                break
            except Exception:
                continue
        return samples
    except Exception as e:
        print(f"[Data] Error: {e}")
        return []