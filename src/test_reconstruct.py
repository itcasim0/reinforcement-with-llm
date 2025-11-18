"""
텍스트 재구성 기능을 테스트하는 스크립트
"""

import json
from pathlib import Path

# 간단한 테스트용 함수
def test_data_loading():
    """데이터 로딩 테스트"""
    json_path = "data/018.논문자료 요약 데이터/01.데이터/1. Training/1. 라벨링데이터_231101_add/training_논문/training_논문/논문요약20231006_0.json"
    
    print(f"Loading data from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터 구조 확인
    if isinstance(data, list) and len(data) > 0:
        papers = data[0].get("data", [])
        print(f"Total papers: {len(papers)}")
        
        # 첫 번째 논문의 구조 확인
        if papers:
            first_paper = papers[0]
            print(f"\n첫 번째 논문 정보:")
            print(f"  제목: {first_paper.get('title')}")
            print(f"  저자: {first_paper.get('author')}")
            print(f"  날짜: {first_paper.get('date')}")
            
            summary_entire = first_paper.get('summary_entire', [])
            print(f"  summary_entire 항목 수: {len(summary_entire)}")
            
            if summary_entire:
                first_entry = summary_entire[0]
                original_text = first_entry.get('orginal_text', '')
                summary_text = first_entry.get('summary_text', '')
                
                print(f"\n첫 번째 summary_entire 항목:")
                print(f"  원본 텍스트 길이: {len(original_text)}")
                print(f"  요약 텍스트 길이: {len(summary_text)}")
                print(f"\n  원본 텍스트 샘플:")
                print(f"  {original_text[:200]}...")
                print(f"\n  요약 텍스트 샘플:")
                print(f"  {summary_text[:200]}...")
                
        return True
    else:
        print("Unexpected data structure")
        return False


def main():
    print("="*60)
    print("데이터 로딩 테스트")
    print("="*60)
    
    try:
        test_data_loading()
        print("\n테스트 완료!")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
