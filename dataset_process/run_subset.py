from processor import DataProcessor

processor = DataProcessor(embedding_model="gemini")
processor.process_dataset(dir_path="./gpt4_dataset_subset")
processor.save_dataset(save_path="./processed_dataset_subset.h5")
print("done")
