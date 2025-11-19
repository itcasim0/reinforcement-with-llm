import json
from typing import List

from src.config.paths import DATA_DIR


class ReconstructDataLoader:
    def __init__(self):
        """
        Initialize the loader.
        Data directory is set relative to ROOT_DIR: data/paper_data/reconstruct
        """
        self.data_dir = DATA_DIR / "paper_data" / "reconstruct"

        if not self.data_dir.exists():
            print(f"Warning: Directory not found: {self.data_dir}")
            self.file_paths = []
        else:
            # Find all json files in the directory
            self.file_paths = sorted(self.data_dir.glob("*.json"))

    def get_reconstructed_text(self) -> List[str]:
        """
        Load all JSON files and extract 'reconstructed_text' values.

        Returns:
            List[str]: A list of all reconstructed texts found in the files.
        """
        reconstructed_texts = []

        for file_path in self.file_paths:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Iterate through results
                    if "results" in data:
                        for result in data["results"]:
                            # Iterate through reconstructed_summaries
                            if "reconstructed_summaries" in result:
                                for summary in result["reconstructed_summaries"]:
                                    if "reconstructed_text" in summary:
                                        reconstructed_texts.append(
                                            summary["reconstructed_text"]
                                        )
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        return reconstructed_texts


if __name__ == "__main__":
    # Example usage
    loader = ReconstructDataLoader()
    texts = loader.get_reconstructed_text()
    print(f"Loaded {len(texts)} reconstructed texts.")
    if texts:
        print(f"First text extracted: {texts[0][:100]}...")
