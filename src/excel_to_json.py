"""
여러 Excel 파일을 하나의 JSON으로 병합
- 핵심 정보만 추출
- 국문 초록 포함된 논문만
"""

import pandas as pd
import json
import glob
import os
from datetime import datetime


def extract_paper_info(row):
    """
    엑셀 행에서 논문 정보 추출
    """
    
    # 기본 정보
    paper = {}
    
    # 제목 (여러 컬럼명 가능)
    for col in ['제목', '논문명', '논문제목', 'title', 'Title']:
        if col in row.index and pd.notna(row[col]):
            paper['title'] = str(row[col]).strip()
            break
    
    # 저자
    for col in ['저자', '저자명', '연구자', 'author', 'Author', '제1저자']:
        if col in row.index and pd.notna(row[col]):
            paper['author'] = str(row[col]).strip()
            break
    
    # 초록 (가장 중요!)
    abstract = ""
    for col in ['초록', '요약', 'abstract', 'Abstract', '국문초록', '국문 초록', '한글초록']:
        if col in row.index and pd.notna(row[col]):
            abstract = str(row[col]).strip()
            if len(abstract) > 50:  # 충분히 긴 초록만
                paper['abstract'] = abstract
                break
    
    # 발행년도
    for col in ['발행년도', '년도', '발간년도', 'year', 'Year', '출판년도']:
        if col in row.index and pd.notna(row[col]):
            try:
                year = int(row[col])
                paper['year'] = year
                break
            except:
                # "2024년" 같은 형식 처리
                import re
                year_match = re.search(r'(\d{4})', str(row[col]))
                if year_match:
                    paper['year'] = int(year_match.group(1))
                break
    
    # 학술지명
    for col in ['학술지명', '학술지', '저널', 'journal', 'Journal', '게재지']:
        if col in row.index and pd.notna(row[col]):
            paper['journal'] = str(row[col]).strip()
            break
    
    # DOI
    for col in ['DOI', 'doi']:
        if col in row.index and pd.notna(row[col]):
            paper['doi'] = str(row[col]).strip()
            break
    
    # 키워드
    for col in ['키워드', 'keyword', 'Keyword', 'keywords', '핵심어']:
        if col in row.index and pd.notna(row[col]):
            paper['keywords'] = str(row[col]).strip()
            break
    
    return paper


def process_excel_files(folder_path=".", output_file="papers_merged.json"):
    """
    폴더 내 모든 Excel 파일 처리
    
    Args:
        folder_path: Excel 파일들이 있는 폴더
        output_file: 출력 JSON 파일명
    """
    
    print("="*70)
    print("📊 Excel → JSON 변환기")
    print("="*70)
    print()
    
    # Excel 파일 찾기
    excel_files = []
    for pattern in ['*.xlsx', '*.xls', '*.csv']:
        excel_files.extend(glob.glob(os.path.join(folder_path, pattern)))
    
    if not excel_files:
        print("❌ Excel 파일을 찾을 수 없습니다!")
        print(f"   폴더: {os.path.abspath(folder_path)}")
        return
    
    print(f"📁 {len(excel_files)}개 파일 발견:")
    for f in excel_files:
        print(f"   - {os.path.basename(f)}")
    print()
    
    # 모든 논문 저장
    all_papers = []
    stats = {
        'total_rows': 0,
        'with_abstract': 0,
        'without_abstract': 0,
        'files_processed': 0
    }
    
    # 각 파일 처리
    for file_path in excel_files:
        print(f"📄 처리 중: {os.path.basename(file_path)}")
        
        try:
            # 파일 읽기
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            elif file_path.endswith('.xls'):
                # .xls 파일은 xlrd 대신 openpyxl 사용
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                except:
                    # openpyxl 실패시 xlrd 시도
                    try:
                        df = pd.read_excel(file_path, engine='xlrd')
                    except:
                        # 둘 다 실패하면 pyxlsb 시도
                        df = pd.read_excel(file_path, engine='pyxlsb')
            else:
                df = pd.read_excel(file_path)
            
            print(f"   ✓ {len(df)}개 행 로드")
            print(f"   ✓ 컬럼: {', '.join(df.columns[:5])}..." if len(df.columns) > 5 else f"   ✓ 컬럼: {', '.join(df.columns)}")
            
            # 각 행 처리
            file_papers = []
            for idx, row in df.iterrows():
                stats['total_rows'] += 1
                
                paper = extract_paper_info(row)
                
                # 초록이 있는 논문만
                if 'abstract' in paper and len(paper.get('abstract', '')) > 50:
                    # 제목도 있어야 함
                    if 'title' in paper and len(paper.get('title', '')) > 5:
                        paper['source_file'] = os.path.basename(file_path)
                        file_papers.append(paper)
                        stats['with_abstract'] += 1
                    else:
                        stats['without_abstract'] += 1
                else:
                    stats['without_abstract'] += 1
            
            all_papers.extend(file_papers)
            stats['files_processed'] += 1
            
            print(f"   ✓ {len(file_papers)}개 논문 추출 (초록 포함)\n")
            
        except Exception as e:
            print(f"   ❌ 오류: {e}\n")
            continue
    
    # 통계
    print("="*70)
    print("📊 변환 결과")
    print("="*70)
    print(f"  처리된 파일: {stats['files_processed']}개")
    print(f"  총 행 수: {stats['total_rows']}개")
    print(f"  초록 있음: {stats['with_abstract']}개 ✓")
    print(f"  초록 없음: {stats['without_abstract']}개 ✗")
    print()
    
    if not all_papers:
        print("⚠️  변환된 논문이 없습니다!")
        return
    
    # 연도별 통계
    years = {}
    for p in all_papers:
        year = p.get('year', 0)
        if year > 0:
            years[year] = years.get(year, 0) + 1
    
    if years:
        print("📅 연도별 분포:")
        for year in sorted(years.keys(), reverse=True):
            print(f"   {year}년: {years[year]}개")
        print()
    
    # JSON 저장
    output = {
        'metadata': {
            'total_papers': len(all_papers),
            'source_files': [os.path.basename(f) for f in excel_files],
            'conversion_date': datetime.now().isoformat(),
            'years': years
        },
        'papers': all_papers
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"💾 저장 완료: {output_file}")
    print(f"   총 {len(all_papers)}개 논문")
    
    # 샘플 출력
    if all_papers:
        print("\n📄 샘플 논문:")
        sample = all_papers[0]
        print(f"   제목: {sample.get('title', 'N/A')[:60]}...")
        print(f"   저자: {sample.get('author', 'N/A')}")
        print(f"   초록: {sample.get('abstract', 'N/A')[:100]}...")
        print(f"   연도: {sample.get('year', 'N/A')}")
    
    print("\n" + "="*70)
    print("✨ 완료!")
    print("="*70)


def main():
    """메인"""
    
    print("Excel → JSON 변환기")
    print()
    
    # 폴더 선택
    folder = input("Excel 파일이 있는 폴더 (기본: 현재 폴더): ").strip() or "."
    
    # 출력 파일명
    output = input("출력 파일명 (기본: papers_merged.json): ").strip() or "papers_merged.json"
    
    print()
    
    # 처리
    process_excel_files(folder_path=folder, output_file=output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  중단됨")
    except Exception as e:
        print(f"\n\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()