from funasr import AutoModel

print("Downloading and caching iic/emotion2vec_plus_base model weights into the container...")
model = AutoModel(model="iic/emotion2vec_plus_base", trust_remote_code=True)
print("Download complete.")
