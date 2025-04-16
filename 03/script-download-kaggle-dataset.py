import kagglehub

# Download latest version
path = kagglehub.dataset_download("vbookshelf/global-wheat-head-dataset-2021")

print("Path to dataset files:", path)