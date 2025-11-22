import arxiv
import numpy as np
from collections import Counter
import re
import json
import time
from typing import List, Dict

class LargeScaleAnalyzer:
    """
    500개 논문 대규모 분석기
    - 다양한 카테고리에서 최신 논문 수집
    - 통계적으로 유의미한 평가 기준 생성
    """
    
    def __init__(self):
        self.papers = []
        self.client = arxiv.Client()
    
    def collect_papers_large_scale(self, target_count=500):
        """
        여러 카테고리에서 500개 논문 수집
        """
        print(f"\n📚 {target_count}개 논문 대규모 수집 시작...")
        print("   카테고리: AI, ML, CV, NLP, Robotics 등\n")
        
        papers = []
        
        # 카테고리별 수집 비율
        categories = {
            'cs.LG': 150,  # Machine Learning
            'cs.CV': 150,  # Computer Vision
            'cs.CL': 100,  # NLP
            'cs.AI': 50,   # AI General
            'cs.RO': 30,   # Robotics
            'cs.NE': 20,   # Neural Computing
        }
        
        for category, count in categories.items():
            print(f"📁 {category} - 목표 {count}개")
            
            try:
                # 최근 2년 논문 (2023-2025)
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=count,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                collected = 0
                for paper in self.client.results(search):
                    # Abstract 길이 필터 (너무 짧은 것 제외)
                    if len(paper.summary.split()) < 50:
                        continue
                    
                    paper_info = {
                        'title': paper.title,
                        'abstract': paper.summary,
                        'year': paper.published.year,
                        'arxiv_id': paper.entry_id.split('/')[-1],
                        'authors': [a.name for a in paper.authors[:3]],
                        'category': category,
                        'type': 'recent_paper'
                    }
                    
                    papers.append(paper_info)
                    collected += 1
                    
                    if collected % 20 == 0:
                        print(f"   진행: {collected}/{count}개")
                    
                    if collected >= count:
                        break
                    
                    time.sleep(0.1)  # Rate limit 방지
                
                print(f"   ✓ 완료: {collected}개 수집\n")
                
            except Exception as e:
                print(f"   ✗ 오류: {type(e).__name__}\n")
        
        self.papers = papers
        
        print(f"\n✅ 총 {len(papers)}개 논문 수집 완료")
        print(f"   연도 범위: {min(p['year'] for p in papers)} ~ {max(p['year'] for p in papers)}")
        
        # 카테고리별 통계
        category_counts = {}
        for p in papers:
            cat = p['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\n📊 카테고리별 분포:")
        for cat, cnt in sorted(category_counts.items()):
            print(f"   {cat}: {cnt}개")
        
        return papers
    
    def analyze_abstracts_comprehensive(self):
        """
        500개 논문 종합 분석
        """
        print("\n" + "="*80)
        print("📊 대규모 논문 초록 종합 분석")
        print("="*80)
        
        if not self.papers:
            print("❌ 분석할 논문이 없습니다")
            return None
        
        abstracts = [p['abstract'] for p in self.papers]
        
        analysis = {
            'metadata': {
                'total_papers': len(self.papers),
                'categories': list(set(p['category'] for p in self.papers)),
                'year_range': [min(p['year'] for p in self.papers), max(p['year'] for p in self.papers)]
            },
            'basic_stats': self._analyze_basic_stats(abstracts),
            'length_distribution': self._analyze_length_distribution(abstracts),
            'structure_patterns': self._analyze_structure_patterns(abstracts),
            'linguistic_features': self._analyze_linguistic_features(abstracts),
            'content_patterns': self._analyze_content_patterns(abstracts),
            'keyword_analysis': self._analyze_keywords(abstracts),
            'sentence_analysis': self._analyze_sentences(abstracts),
            'advanced_metrics': self._analyze_advanced_metrics(abstracts),
            'category_comparison': self._analyze_by_category()
        }
        
        self._print_comprehensive_analysis(analysis)
        
        return analysis
    
    def _analyze_basic_stats(self, abstracts):
        """기본 통계 (500개 기준)"""
        word_counts = [len(abs.split()) for abs in abstracts]
        char_counts = [len(abs) for abs in abstracts]
        
        return {
            'total_papers': len(abstracts),
            'word_count': {
                'mean': float(np.mean(word_counts)),
                'std': float(np.std(word_counts)),
                'min': int(np.min(word_counts)),
                'max': int(np.max(word_counts)),
                'median': float(np.median(word_counts)),
                'q25': float(np.percentile(word_counts, 25)),
                'q50': float(np.percentile(word_counts, 50)),
                'q75': float(np.percentile(word_counts, 75)),
                'q90': float(np.percentile(word_counts, 90)),
                'q10': float(np.percentile(word_counts, 10))
            },
            'char_count': {
                'mean': float(np.mean(char_counts)),
                'median': float(np.median(char_counts))
            }
        }
    
    def _analyze_length_distribution(self, abstracts):
        """길이 분포 (더 세분화)"""
        word_counts = [len(abs.split()) for abs in abstracts]
        
        bins = {
            'very_short': sum(1 for w in word_counts if w < 100),
            'short': sum(1 for w in word_counts if 100 <= w < 150),
            'medium': sum(1 for w in word_counts if 150 <= w < 200),
            'long': sum(1 for w in word_counts if 200 <= w < 250),
            'very_long': sum(1 for w in word_counts if w >= 250)
        }
        
        total = len(abstracts)
        distribution = {k: float(v / total) for k, v in bins.items()}
        
        # 히스토그램 데이터 추가
        hist, bin_edges = np.histogram(word_counts, bins=20)
        distribution['histogram'] = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
        return distribution
    
    def _analyze_structure_patterns(self, abstracts):
        """구조 패턴 (500개 분석)"""
        patterns = {
            'has_background': 0,
            'has_problem': 0,
            'has_method': 0,
            'has_results': 0,
            'has_conclusion': 0,
            'full_structure': 0,
            'starts_with_context': 0,
            'ends_with_impact': 0,
        }
        
        background_kw = ['existing', 'current', 'previous', 'traditional', 'conventional']
        problem_kw = ['however', 'challenge', 'difficult', 'limitation', 'problem']
        method_kw = ['we propose', 'we present', 'we introduce', 'our approach', 'our method', 'we develop', 'our model']
        result_kw = ['achieve', 'outperform', 'demonstrate', 'show that', 'significantly', 'improvement', 'better']
        conclusion_kw = ['therefore', 'thus', 'overall', 'in summary', 'our work']
        
        for abstract in abstracts:
            sentences = [s.strip() for s in re.split(r'[.!?]+', abstract) if s.strip()]
            if len(sentences) < 3:
                continue
            
            abs_lower = abstract.lower()
            first_sent = sentences[0].lower()
            last_sent = sentences[-1].lower()
            
            has_bg = any(kw in abs_lower for kw in background_kw)
            has_prob = any(kw in abs_lower for kw in problem_kw)
            has_meth = any(kw in abs_lower for kw in method_kw)
            has_res = any(kw in abs_lower for kw in result_kw)
            has_conc = any(kw in abs_lower for kw in conclusion_kw)
            
            if has_bg:
                patterns['has_background'] += 1
            if has_prob:
                patterns['has_problem'] += 1
            if has_meth:
                patterns['has_method'] += 1
            if has_res:
                patterns['has_results'] += 1
            if has_conc:
                patterns['has_conclusion'] += 1
            if has_bg and has_prob and has_meth and has_res:
                patterns['full_structure'] += 1
            
            if any(kw in first_sent for kw in background_kw + ['recent', 'many', 'in']):
                patterns['starts_with_context'] += 1
            
            if any(kw in last_sent for kw in result_kw + conclusion_kw):
                patterns['ends_with_impact'] += 1
        
        total = len(abstracts)
        return {k: float(v / total) for k, v in patterns.items()}
    
    def _analyze_linguistic_features(self, abstracts):
        """언어학적 특징 (500개)"""
        features = {
            'has_numbers': 0,
            'has_percentage': 0,
            'has_equation': 0,
            'has_comparison': 0,
            'uses_we': 0,
            'uses_our': 0,
            'uses_this_paper': 0,
            'passive_voice': 0,
            'active_voice': 0,
            'has_parentheses': 0,
            'has_hyphen': 0,
            'has_colon': 0,
            'avg_commas': 0,
        }
        
        total_commas = 0
        
        for abstract in abstracts:
            abs_lower = abstract.lower()
            
            if re.search(r'\d+', abstract):
                features['has_numbers'] += 1
            if '%' in abstract or 'percent' in abs_lower:
                features['has_percentage'] += 1
            if any(x in abstract for x in ['$', '\\', 'equation']):
                features['has_equation'] += 1
            if any(w in abs_lower for w in ['better than', 'compared to', 'outperform', 'superior', 'vs']):
                features['has_comparison'] += 1
            if ' we ' in abs_lower or abs_lower.startswith('we '):
                features['uses_we'] += 1
            if ' our ' in abs_lower:
                features['uses_our'] += 1
            if 'this paper' in abs_lower or 'this work' in abs_lower:
                features['uses_this_paper'] += 1
            if any(w in abs_lower for w in [' is ', ' are ', ' was ', ' were ']):
                features['passive_voice'] += 1
            if any(w in abs_lower for w in [' we ', ' they ', ' it ']):
                features['active_voice'] += 1
            if '(' in abstract:
                features['has_parentheses'] += 1
            if '-' in abstract:
                features['has_hyphen'] += 1
            if ':' in abstract:
                features['has_colon'] += 1
            
            total_commas += abstract.count(',')
        
        total = len(abstracts)
        result = {k: float(v / total) for k, v in features.items()}
        result['avg_commas'] = float(total_commas / total)
        
        return result
    
    def _analyze_content_patterns(self, abstracts):
        """내용 패턴 (500개)"""
        patterns = {
            'mentions_sota': 0,
            'mentions_dataset': 0,
            'mentions_benchmark': 0,
            'mentions_architecture': 0,
            'mentions_model': 0,
            'mentions_novel': 0,
            'mentions_evaluation': 0,
            'mentions_training': 0,
            'mentions_performance': 0,
            'mentions_efficiency': 0,
            'mentions_scalability': 0,
            'mentions_real_world': 0,
        }
        
        datasets = ['imagenet', 'coco', 'mnist', 'cifar', 'glue', 'squad', 'dataset']
        
        for abstract in abstracts:
            abs_lower = abstract.lower()
            
            if 'state-of-the-art' in abs_lower or 'sota' in abs_lower:
                patterns['mentions_sota'] += 1
            if any(ds in abs_lower for ds in datasets):
                patterns['mentions_dataset'] += 1
            if 'benchmark' in abs_lower:
                patterns['mentions_benchmark'] += 1
            if 'architecture' in abs_lower:
                patterns['mentions_architecture'] += 1
            if 'model' in abs_lower:
                patterns['mentions_model'] += 1
            if 'novel' in abs_lower:
                patterns['mentions_novel'] += 1
            if any(w in abs_lower for w in ['evaluation', 'experiment', 'evaluate']):
                patterns['mentions_evaluation'] += 1
            if 'training' in abs_lower or 'train' in abs_lower:
                patterns['mentions_training'] += 1
            if 'performance' in abs_lower or 'accuracy' in abs_lower:
                patterns['mentions_performance'] += 1
            if 'efficient' in abs_lower or 'efficiency' in abs_lower:
                patterns['mentions_efficiency'] += 1
            if 'scalable' in abs_lower or 'scalability' in abs_lower:
                patterns['mentions_scalability'] += 1
            if 'real-world' in abs_lower or 'practical' in abs_lower:
                patterns['mentions_real_world'] += 1
        
        total = len(abstracts)
        return {k: float(v / total) for k, v in patterns.items()}
    
    def _analyze_keywords(self, abstracts):
        """키워드 빈도 (500개 - 상위 50개)"""
        all_words = []
        for abstract in abstracts:
            words = re.findall(r'\b[a-z]{4,}\b', abstract.lower())
            all_words.extend(words)
        
        stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'which', 'their', 
                    'these', 'such', 'than', 'also', 'more', 'other', 'into', 'only',
                    'over', 'very', 'when', 'them', 'about', 'both', 'most', 'many',
                    'where', 'while', 'then', 'there', 'here', 'each', 'some'}
        filtered = [w for w in all_words if w not in stopwords]
        
        counter = Counter(filtered)
        return dict(counter.most_common(50))
    
    def _analyze_sentences(self, abstracts):
        """문장 분석 (500개)"""
        all_sentence_lengths = []
        sentence_counts = []
        first_sentence_lengths = []
        last_sentence_lengths = []
        
        for abstract in abstracts:
            sentences = [s.strip() for s in re.split(r'[.!?]+', abstract) if s.strip()]
            sentence_counts.append(len(sentences))
            
            for i, sent in enumerate(sentences):
                words = len(sent.split())
                all_sentence_lengths.append(words)
                
                if i == 0:
                    first_sentence_lengths.append(words)
                if i == len(sentences) - 1:
                    last_sentence_lengths.append(words)
        
        return {
            'avg_sentences_per_abstract': float(np.mean(sentence_counts)),
            'median_sentences_per_abstract': float(np.median(sentence_counts)),
            'avg_words_per_sentence': float(np.mean(all_sentence_lengths)),
            'median_words_per_sentence': float(np.median(all_sentence_lengths)),
            'sentence_length_std': float(np.std(all_sentence_lengths)),
            'first_sentence_avg': float(np.mean(first_sentence_lengths)),
            'last_sentence_avg': float(np.mean(last_sentence_lengths)),
        }
    
    def _analyze_advanced_metrics(self, abstracts):
        """고급 지표"""
        metrics = {
            'lexical_diversity': [],
            'technical_term_density': [],
            'readability_scores': []
        }
        
        for abstract in abstracts:
            words = abstract.lower().split()
            unique_words = set(words)
            
            # Lexical Diversity (Type-Token Ratio)
            if len(words) > 0:
                ttx = len(unique_words) / len(words)
                metrics['lexical_diversity'].append(ttx)
            
            # Technical Term Density (긴 단어 비율)
            technical = sum(1 for w in words if len(w) >= 8)
            if len(words) > 0:
                metrics['technical_term_density'].append(technical / len(words))
            
            # 간단한 가독성 점수
            sentences = [s.strip() for s in re.split(r'[.!?]+', abstract) if s.strip()]
            if len(sentences) > 0 and len(words) > 0:
                avg_sent_len = len(words) / len(sentences)
                long_words = sum(1 for w in words if len(w) >= 7)
                readability = avg_sent_len + (long_words / len(words) * 100)
                metrics['readability_scores'].append(readability)
        
        return {
            'avg_lexical_diversity': float(np.mean(metrics['lexical_diversity'])),
            'avg_technical_density': float(np.mean(metrics['technical_term_density'])),
            'avg_readability': float(np.mean(metrics['readability_scores']))
        }
    
    def _analyze_by_category(self):
        """카테고리별 비교 분석"""
        category_stats = {}
        
        for category in set(p['category'] for p in self.papers):
            cat_papers = [p for p in self.papers if p['category'] == category]
            cat_abstracts = [p['abstract'] for p in cat_papers]
            
            if not cat_abstracts:
                continue
            
            word_counts = [len(abs.split()) for abs in cat_abstracts]
            
            category_stats[category] = {
                'count': len(cat_papers),
                'avg_length': float(np.mean(word_counts)),
                'median_length': float(np.median(word_counts))
            }
        
        return category_stats
    
    def _print_comprehensive_analysis(self, analysis):
        """종합 분석 결과 출력"""
        print("\n" + "="*80)
        print("📈 기본 통계 (500개 논문)")
        print("="*80)
        
        stats = analysis['basic_stats']
        print(f"총 논문: {stats['total_papers']}개")
        print(f"\n단어 수 통계:")
        print(f"  평균: {stats['word_count']['mean']:.1f} ± {stats['word_count']['std']:.1f}")
        print(f"  중앙값: {stats['word_count']['median']:.1f}")
        print(f"  범위: {stats['word_count']['min']} ~ {stats['word_count']['max']}")
        print(f"  Q10-Q90: {stats['word_count']['q10']:.0f} ~ {stats['word_count']['q90']:.0f}")
        print(f"  Q25-Q75: {stats['word_count']['q25']:.0f} ~ {stats['word_count']['q75']:.0f}")
        print(f"\n👉 권장 길이: {stats['word_count']['q25']:.0f}-{stats['word_count']['q75']:.0f} 단어")
        print(f"👉 최적 길이: {stats['word_count']['median']:.0f} 단어")
        
        print("\n" + "="*80)
        print("📊 길이 분포")
        print("="*80)
        dist = analysis['length_distribution']
        print(f"  매우 짧음 (<100):     {dist['very_short']*100:5.1f}%  {'█' * int(dist['very_short']*50)}")
        print(f"  짧음 (100-150):       {dist['short']*100:5.1f}%  {'█' * int(dist['short']*50)}")
        print(f"  중간 (150-200):       {dist['medium']*100:5.1f}%  {'█' * int(dist['medium']*50)}")
        print(f"  김 (200-250):         {dist['long']*100:5.1f}%  {'█' * int(dist['long']*50)}")
        print(f"  매우 김 (250+):       {dist['very_long']*100:5.1f}%  {'█' * int(dist['very_long']*50)}")
        
        print("\n" + "="*80)
        print("🏗️  구조 패턴 (500개 분석)")
        print("="*80)
        struct = analysis['structure_patterns']
        print(f"배경/맥락:             {struct['has_background']*100:5.1f}%")
        print(f"문제 정의:             {struct['has_problem']*100:5.1f}%")
        print(f"방법론 제시:           {struct['has_method']*100:5.1f}%")
        print(f"결과 제시:             {struct['has_results']*100:5.1f}%")
        print(f"결론:                  {struct['has_conclusion']*100:5.1f}%")
        print(f"완전한 구조:           {struct['full_structure']*100:5.1f}%")
        
        print("\n" + "="*80)
        print("✍️  언어 특징 (500개)")
        print("="*80)
        ling = analysis['linguistic_features']
        print(f"숫자 포함:             {ling['has_numbers']*100:5.1f}%")
        print(f"백분율 사용:           {ling['has_percentage']*100:5.1f}%")
        print(f"비교 표현:             {ling['has_comparison']*100:5.1f}%")
        print(f"'We' 사용:             {ling['uses_we']*100:5.1f}%")
        print(f"'Our' 사용:            {ling['uses_our']*100:5.1f}%")
        print(f"'This paper' 사용:     {ling['uses_this_paper']*100:5.1f}%")
        print(f"평균 쉼표 개수:        {ling['avg_commas']:.1f}개")
        
        print("\n" + "="*80)
        print("📝 내용 패턴 (500개)")
        print("="*80)
        content = analysis['content_patterns']
        print(f"SOTA 언급:             {content['mentions_sota']*100:5.1f}%")
        print(f"데이터셋:              {content['mentions_dataset']*100:5.1f}%")
        print(f"벤치마크:              {content['mentions_benchmark']*100:5.1f}%")
        print(f"모델 언급:             {content['mentions_model']*100:5.1f}%")
        print(f"Novel 강조:            {content['mentions_novel']*100:5.1f}%")
        print(f"평가/실험:             {content['mentions_evaluation']*100:5.1f}%")
        print(f"성능:                  {content['mentions_performance']*100:5.1f}%")
        print(f"효율성:                {content['mentions_efficiency']*100:5.1f}%")
        
        print("\n" + "="*80)
        print("📄 문장 분석")
        print("="*80)
        sent = analysis['sentence_analysis']
        print(f"초록당 평균 문장:      {sent['avg_sentences_per_abstract']:.1f}개")
        print(f"문장당 평균 단어:      {sent['avg_words_per_sentence']:.1f}개")
        print(f"문장당 중앙값:         {sent['median_words_per_sentence']:.1f}개")
        print(f"첫 문장 평균:          {sent['first_sentence_avg']:.1f}개")
        print(f"마지막 문장 평균:      {sent['last_sentence_avg']:.1f}개")
        
        print("\n" + "="*80)
        print("🔬 고급 지표")
        print("="*80)
        adv = analysis['advanced_metrics']
        print(f"어휘 다양성 (TTR):     {adv['avg_lexical_diversity']:.3f}")
        print(f"전문 용어 밀도:        {adv['avg_technical_density']:.3f}")
        print(f"가독성 점수:           {adv['avg_readability']:.1f}")
        
        print("\n" + "="*80)
        print("🔤 상위 30개 핵심 키워드")
        print("="*80)
        keywords = analysis['keyword_analysis']
        for i, (word, count) in enumerate(list(keywords.items())[:30], 1):
            if i % 3 == 1:
                print()
            print(f"{word:18s}({count:4d})", end=" ")
        print("\n")
        
        print("="*80)
        print("📂 카테고리별 비교")
        print("="*80)
        for cat, stats in sorted(analysis['category_comparison'].items()):
            print(f"{cat:10s} | {stats['count']:3d}개 | 평균 {stats['avg_length']:.0f}단어")
    
    def save_results(self, filename='large_scale_500_analysis.json'):
        """결과 저장"""
        analysis = self.analyze_abstracts_comprehensive()
        
        if analysis is None:
            return None
        
        output = {
            'metadata': {
                'total_papers': len(self.papers),
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'large_scale_multi_category',
                'categories': list(set(p['category'] for p in self.papers))
            },
            'analysis': analysis,
            'papers_sample': self.papers[:20]  # 샘플만 저장 (파일 크기 관리)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 저장 완료: {filename}")
        print(f"   - {len(self.papers)}개 논문 분석 결과")
        print(f"   - 통계적 평가 기준 포함")
        
        return output
    
    def create_evaluation_model(self, analysis):
        """통계 기반 평가 모델 생성"""
        print("\n" + "="*80)
        print("✅ 평가 모델 생성 (500개 기반)")
        print("="*80)
        
        stats = analysis['basic_stats']
        struct = analysis['structure_patterns']
        ling = analysis['linguistic_features']
        content = analysis['content_patterns']
        sent = analysis['sentence_analysis']
        
        model = {
            'scoring_criteria': {
                'length_score': {
                    'weight': 0.15,
                    'optimal_min': stats['word_count']['q25'],
                    'optimal_max': stats['word_count']['q75'],
                    'target': stats['word_count']['median'],
                    'method': 'gaussian_penalty'
                },
                'structure_score': {
                    'weight': 0.30,
                    'has_background': struct['has_background'],
                    'has_method': struct['has_method'],
                    'has_results': struct['has_results'],
                    'full_structure_bonus': 0.2
                },
                'linguistic_score': {
                    'weight': 0.20,
                    'has_numbers': ling['has_numbers'],
                    'has_comparison': ling['has_comparison'],
                    'first_person': max(ling['uses_we'], ling['uses_our']),
                    'target_commas': ling['avg_commas']
                },
                'content_score': {
                    'weight': 0.20,
                    'evaluation': content['mentions_evaluation'],
                    'dataset': content['mentions_dataset'],
                    'performance': content['mentions_performance'],
                    'model': content['mentions_model']
                },
                'sentence_score': {
                    'weight': 0.15,
                    'target_words_per_sentence': sent['avg_words_per_sentence'],
                    'acceptable_range': [
                        sent['avg_words_per_sentence'] - 5,
                        sent['avg_words_per_sentence'] + 5
                    ]
                }
            },
            'thresholds': {
                'excellent': 0.85,
                'good': 0.70,
                'acceptable': 0.55,
                'poor': 0.40
            },
            'statistics': {
                'based_on': len(self.papers),
                'word_count_distribution': {
                    'q10': stats['word_count']['q10'],
                    'q25': stats['word_count']['q25'],
                    'q50': stats['word_count']['q50'],
                    'q75': stats['word_count']['q75'],
                    'q90': stats['word_count']['q90']
                }
            }
        }
        
        print("\n📊 평가 모델 요약:")
        print(f"\n1. 길이 점수 (15%):")
        print(f"   최적: {model['scoring_criteria']['length_score']['optimal_min']:.0f}-{model['scoring_criteria']['length_score']['optimal_max']:.0f} 단어")
        print(f"   목표: {model['scoring_criteria']['length_score']['target']:.0f} 단어")
        
        print(f"\n2. 구조 점수 (30%):")
        print(f"   배경: {struct['has_background']*100:.0f}%")
        print(f"   방법: {struct['has_method']*100:.0f}%")
        print(f"   결과: {struct['has_results']*100:.0f}%")
        
        print(f"\n3. 언어 점수 (20%):")
        print(f"   숫자: {ling['has_numbers']*100:.0f}%")
        print(f"   비교: {ling['has_comparison']*100:.0f}%")
        
        print(f"\n4. 내용 점수 (20%):")
        print(f"   평가: {content['mentions_evaluation']*100:.0f}%")
        print(f"   성능: {content['mentions_performance']*100:.0f}%")
        
        print(f"\n5. 문장 점수 (15%):")
        print(f"   목표: {sent['avg_words_per_sentence']:.1f} 단어/문장")
        
        print(f"\n🎯 등급 기준:")
        print(f"   Excellent: {model['thresholds']['excellent']*100:.0f}% 이상")
        print(f"   Good:      {model['thresholds']['good']*100:.0f}% 이상")
        print(f"   Acceptable: {model['thresholds']['acceptable']*100:.0f}% 이상")
        
        return model


if __name__ == "__main__":
    print("="*80)
    print("🎯 500개 논문 대규모 분석 및 평가 모델 생성")
    print("="*80)
    
    analyzer = LargeScaleAnalyzer()
    
    # 500개 논문 수집
    papers = analyzer.collect_papers_large_scale(target_count=500)
    
    if len(papers) >= 100:  # 최소 100개 이상
        # 분석
        output = analyzer.save_results('large_scale_500_analysis.json')
        
        # 평가 모델 생성
        if output:
            model = analyzer.create_evaluation_model(output['analysis'])
            
            with open('evaluation_model_500.json', 'w', encoding='utf-8') as f:
                json.dump(model, f, indent=2, ensure_ascii=False)
            print("\n💾 평가 모델 저장: evaluation_model_500.json")
            
            print("\n" + "="*80)
            print("✅ 완료!")
            print("="*80)
            print(f"수집: {len(papers)}개 논문")
            print(f"분석: large_scale_500_analysis.json")
            print(f"모델: evaluation_model_500.json")
    else:
        print("\n❌ 논문 수집 실패 (최소 100개 필요)")