from typing import List

from config.paths import DATA_DIR

from environments.editing_env.components import Document

from utils.util import load_json


class ReconstructDataLoader:
    def __init__(self):
        """
        Initialize the loader.
        Data directory is set relative to ROOT_DIR: data/paper_data/reconstruct
        """
        self.data_dir = DATA_DIR / "paper_data" / "reconstruct"

        if not self.data_dir.exists():
            print(f"[WARN] Directory not found: {self.data_dir}")
            self.file_paths = []
        else:
            # Find all json files in the directory
            self.file_paths = sorted(self.data_dir.glob("*.json"))

    def get_reconstructed_text(self, max_docs=float("inf")) -> List[Document]:
        """
        Load all JSON files and extract 'reconstructed_text' values.

        Returns:
            List[str]: A list of all reconstructed texts found in the files.
        """

        reconstructed_texts = []

        for file_path in self.file_paths:
            try:
                data = load_json(file_path)

                results: List[dict] = data.get("results", [])
                for result in results:
                    recon_summary: List[dict] = result.get(
                        "reconstructed_summaries", []
                    )
                    for summary in recon_summary:
                        text = summary.get("reconstructed_text")
                        if text:
                            reconstructed_texts.append(Document(text=text))
                            if len(reconstructed_texts) > max_docs:
                                break

            except Exception as e:
                print(f"[WARN] Error reading {file_path}: {e}")
                continue

        return reconstructed_texts


if __name__ == "__main__":
    # Example usage
    loader = ReconstructDataLoader()
    texts = loader.get_reconstructed_text()
    print(f"Loaded {len(texts)} reconstructed texts.")
    if texts:
        print(f"First text extracted: {texts[0].text[:100]}...")
