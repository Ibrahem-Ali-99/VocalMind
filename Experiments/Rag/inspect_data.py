from datasets import load_from_disk

try:
    ds = load_from_disk("c:/Users/PC/Desktop/VocalMind/Experiments/Rag/data/ragbench")
    print("Dataset keys:", ds.keys())

    # Inspect first entry of 'train'
    if "train" in ds:
        print("\nStructure of first item in 'train':")
        # clear large content for display
        item = ds["train"][0]
        # Summarize large fields
        for k, v in item.items():
            if isinstance(v, str) and len(v) > 100:
                print(f"  {k}: <str length {len(v)}>")
            else:
                print(f"  {k}: {v}")

except Exception as e:
    print(f"Error: {e}")
