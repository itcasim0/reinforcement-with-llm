from typing import List

from config.paths import DATA_DIR

from environments.editing_env.components.component import Document

from utils.util import load_json


class DomesticReconstructDataLoader:
    def __init__(self):
        """
        국내 논문 초록을 재가공한 데이터 클래스를 별도로 하나 더 만든 이유는,
        JSON 파일 파싱하는 방법이 조금 다르고, 로드하는 파일 개수가 조금 달라서 우선 클래스 하나 만듦
        """

        # 국내 논문 초록 데이터 파일은 현재 하나이니까 명시적으로 선언
        self.data_path = (
            DATA_DIR
            / "paper_data"
            / "reconstruct"
            / "paper_abstract_20251125_002418.json"
        )

    def get_reconstructed_text(self, max_docs=float("inf")) -> List[Document]:

        reconstructed_texts = []

        try:
            data = load_json(self.data_path)

            results: List[dict] = data.get("results", [])
            for result in results:
                abstract_text = result.get("abstract_reconstructed", "")
                if abstract_text:
                    reconstructed_texts.append(Document(text=abstract_text))
                    if len(reconstructed_texts) > max_docs:
                        break

        except Exception as e:
            print(f"[WARN] Error reading {self.data_path}: {e}")

        return reconstructed_texts


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

class NoiseDataLoader:
    def __init__(self):
        """
        맞춤법 및 띄어쓰기 오류 데이터
        """

        # 국내 논문 초록 데이터 파일은 현재 하나이니까 명시적으로 선언
        self.data_path = (
            DATA_DIR
            / "paper_data"
            / "noise"
            / "paper_abstract_with_noise_20251125_002418.json"
        )
    
    def get_noise_text(self, max_docs=float("inf")) -> List[Document]:

        noise_texts = []

        try:
            data = load_json(self.data_path)

            results: List[dict] = data.get("results", [])
            for result in results:
                abstract_text = result.get("abstract_noise", "")
                if abstract_text:
                    noise_texts.append(Document(text=abstract_text))
                    if len(noise_texts) > max_docs:
                        break

        except Exception as e:
            print(f"[WARN] Error reading {self.data_path}: {e}")

        return noise_texts

if __name__ == "__main__":
    # Example usage
    loader = ReconstructDataLoader()
    texts = loader.get_reconstructed_text()
    print(f"Loaded {len(texts)} reconstructed texts.")
    if texts:
        print(f"First text extracted: {texts[0].text[:100]}...")
