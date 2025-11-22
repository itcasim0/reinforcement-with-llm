import re
import numpy as np
from typing import Dict, List
from collections import Counter

class AbstractQualityEvaluator:
    """논문 초록의 품질을 객관적으로 평가하는 클래스 (수정 버전)"""
    
    def __init__(self):
        self.structure_keywords = {
            'background': ['background', 'context', 'motivation', 'problem', 'challenge'],
            'objective': ['aim', 'goal', 'objective', 'purpose', 'investigate', 'study', 'explore'],
            'method': ['method', 'approach', 'technique', 'algorithm', 'framework', 'model', 'propose'],
            'result': ['result', 'finding', 'demonstrate', 'show', 'achieve', 'performance'],
            'conclusion': ['conclude', 'implication', 'suggest', 'contribution', 'significant']
        }
        
        self.academic_connectives = [
            'however', 'moreover', 'furthermore', 'therefore', 'thus',
            'consequently', 'nevertheless', 'additionally', 'specifically'
        ]
    
    def evaluate_structure_completeness(self, abstract: str) -> Dict[str, float]:
        """구조적 완성도 평가"""
        abstract_lower = abstract.lower()
        scores = {}
        
        for section, keywords in self.structure_keywords.items():
            keyword_count = sum(1 for kw in keywords if kw in abstract_lower)
            scores[section] = min(1.0, keyword_count / len(keywords))
        
        structure_completeness = np.mean(list(scores.values()))
        
        return {
            'section_scores': scores,
            'structure_completeness': structure_completeness,
            'missing_sections': [k for k, v in scores.items() if v < 0.2]
        }
    
    def evaluate_length(self, abstract: str) -> Dict[str, float]:
        """길이 평가"""
        words = abstract.split()
        word_count = len(words)
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', abstract) if s.strip()]
        sentence_count = len(sentences)
        
        # 길이 점수
        if 150 <= word_count <= 300:
            length_score = 1.0
        elif word_count < 150:
            length_score = word_count / 150
        else:
            length_score = max(0, 1.0 - (word_count - 300) / 300)
        
        # 문장 길이 점수
        avg_words_per_sentence = word_count / max(1, sentence_count)
        
        if 15 <= avg_words_per_sentence <= 25:
            sentence_length_score = 1.0
        else:
            sentence_length_score = max(0, 1.0 - abs(avg_words_per_sentence - 20) / 20)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': avg_words_per_sentence,
            'length_score': length_score,
            'sentence_length_score': sentence_length_score,
            'overall_length_score': (length_score + sentence_length_score) / 2
        }
    
    def evaluate_academic_style(self, abstract: str) -> Dict[str, float]:
        """학술적 표현 평가"""
        abstract_lower = abstract.lower()
        
        # 연결어
        connective_count = sum(1 for conn in self.academic_connectives if conn in abstract_lower)
        connective_score = min(1.0, connective_count / 3)
        
        # 1인칭 사용
        first_person = ['i ', 'we ', 'our ', 'my ', 'me ']
        first_person_count = sum(abstract_lower.count(fp) for fp in first_person)
        first_person_penalty = max(0, 1.0 - first_person_count * 0.2)
        
        # 수동태
        passive_indicators = ['is ', 'are ', 'was ', 'were ', 'been ', 'being ']
        passive_count = sum(abstract_lower.count(pi) for pi in passive_indicators)
        passive_score = min(1.0, passive_count / 5)
        
        academic_score = (connective_score + first_person_penalty + passive_score) / 3
        
        return {
            'connective_count': connective_count,
            'connective_score': connective_score,
            'first_person_count': first_person_count,
            'first_person_penalty': first_person_penalty,
            'passive_score': passive_score,
            'academic_style_score': academic_score
        }
    
    def evaluate_information_density(self, abstract: str) -> Dict[str, float]:
        """정보 밀도 평가"""
        words = abstract.split()
        word_count = len(words)
        
        if word_count == 0:
            return {'information_density_score': 0.0}
        
        # 숫자/통계
        numbers = re.findall(r'\d+\.?\d*%?', abstract)
        has_numbers = len(numbers) > 0
        number_score = 1.0 if has_numbers else 0.5
        
        # 전문 용어
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:-[A-Za-z]+)*\b', abstract)
        technical_density = len(technical_terms) / word_count
        technical_score = min(1.0, technical_density * 10)
        
        # 불필요한 수식어
        filler_words = ['very', 'really', 'quite', 'somewhat', 'rather']
        filler_count = sum(abstract.lower().count(f' {fw} ') for fw in filler_words)
        filler_penalty = max(0, 1.0 - filler_count * 0.3)
        
        density_score = (number_score + technical_score + filler_penalty) / 3
        
        return {
            'has_numbers': has_numbers,
            'number_count': len(numbers),
            'number_score': number_score,
            'technical_term_count': len(technical_terms),
            'technical_score': technical_score,
            'filler_count': filler_count,
            'filler_penalty': filler_penalty,
            'information_density_score': density_score
        }
    
    def evaluate_clarity(self, abstract: str) -> Dict[str, float]:
        """명확성 평가"""
        abstract_lower = abstract.lower()
        
        # 모호한 표현
        vague_terms = ['various', 'several', 'many', 'some', 'few', 'often', 'sometimes']
        vague_count = sum(abstract_lower.count(f' {vt} ') for vt in vague_terms)
        vague_penalty = max(0, 1.0 - vague_count * 0.15)
        
        # 약어 설명
        abbreviations_with_explanation = re.findall(r'\b([A-Z]{2,})\s*\([^)]+\)', abstract)
        abbreviations_without = re.findall(r'\b[A-Z]{2,}\b', abstract)
        
        if len(abbreviations_without) > 0:
            abbrev_explanation_score = len(abbreviations_with_explanation) / len(set(abbreviations_without))
        else:
            abbrev_explanation_score = 1.0
        
        clarity_score = (vague_penalty + abbrev_explanation_score) / 2
        
        return {
            'vague_count': vague_count,
            'vague_penalty': vague_penalty,
            'abbreviations_explained': len(abbreviations_with_explanation),
            'abbreviations_total': len(set(abbreviations_without)),
            'abbrev_explanation_score': abbrev_explanation_score,
            'clarity_score': clarity_score
        }
    
    def evaluate_coherence(self, abstract: str) -> Dict[str, float]:
        """일관성 평가"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', abstract) if s.strip()]
        
        if len(sentences) < 2:
            return {'coherence_score': 0.5}
        
        # 연결어 사용
        transition_count = 0
        for sent in sentences[1:]:
            sent_lower = sent.lower()
            if any(conn in sent_lower for conn in self.academic_connectives):
                transition_count += 1
        
        transition_score = min(1.0, transition_count / (len(sentences) - 1))
        
        # 반복 개념
        all_words = []
        for sent in sentences:
            words = [w.lower() for w in re.findall(r'\b\w+\b', sent) if len(w) > 4]
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        repeated_concepts = [w for w, c in word_freq.items() if c >= 2]
        concept_consistency = min(1.0, len(repeated_concepts) / 5)
        
        coherence_score = (transition_score + concept_consistency) / 2
        
        return {
            'transition_count': transition_count,
            'transition_score': transition_score,
            'repeated_concepts': len(repeated_concepts),
            'concept_consistency': concept_consistency,
            'coherence_score': coherence_score
        }
    
    def evaluate_abstract(self, abstract: str) -> Dict:
        """종합 평가"""
        structure = self.evaluate_structure_completeness(abstract)
        length = self.evaluate_length(abstract)
        academic = self.evaluate_academic_style(abstract)
        density = self.evaluate_information_density(abstract)
        clarity = self.evaluate_clarity(abstract)
        coherence = self.evaluate_coherence(abstract)
        
        weights = {
            'structure': 0.25,
            'length': 0.10,
            'academic': 0.15,
            'density': 0.20,
            'clarity': 0.15,
            'coherence': 0.15
        }
        
        overall_score = (
            structure['structure_completeness'] * weights['structure'] +
            length['overall_length_score'] * weights['length'] +
            academic['academic_style_score'] * weights['academic'] +
            density['information_density_score'] * weights['density'] +
            clarity['clarity_score'] * weights['clarity'] +
            coherence['coherence_score'] * weights['coherence']
        )
        
        return {
            'overall_score': overall_score,
            'structure': structure,
            'length': length,
            'academic_style': academic,
            'information_density': density,
            'clarity': clarity,
            'coherence': coherence,
            'grade': self._score_to_grade(overall_score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= 0.9:
            return 'A+ (Excellent)'
        elif score >= 0.8:
            return 'A (Very Good)'
        elif score >= 0.7:
            return 'B (Good)'
        elif score >= 0.6:
            return 'C (Acceptable)'
        else:
            return 'D (Needs Improvement)'


if __name__ == "__main__":
    evaluator = AbstractQualityEvaluator()
    
    sample_abstract = """
    Deep learning has revolutionized computer vision, but training large models requires 
    substantial computational resources. This study proposes an efficient training framework 
    that reduces GPU memory usage by 40% while maintaining model accuracy. Our approach 
    combines gradient checkpointing with mixed-precision training, achieving 1.5x speedup 
    compared to baseline methods. Experiments on ImageNet demonstrate that our method reaches 
    76.3% top-1 accuracy, comparable to standard training. These findings suggest that 
    efficient training techniques can significantly reduce costs without sacrificing performance.
    """
    
    results = evaluator.evaluate_abstract(sample_abstract)
    
    print(f"\n{'='*60}")
    print(f"Overall Score: {results['overall_score']:.3f} - {results['grade']}")
    print(f"{'='*60}\n")
    
    print("Detailed Scores:")
    print(f"  Structure: {results['structure']['structure_completeness']:.3f}")
    print(f"    Missing: {results['structure']['missing_sections']}")
    
    print(f"\n  Length: {results['length']['overall_length_score']:.3f}")
    print(f"    Words: {results['length']['word_count']}")
    print(f"    Sentences: {results['length']['sentence_count']}")
    
    print(f"\n  Academic Style: {results['academic_style']['academic_style_score']:.3f}")
    print(f"    Connectives: {results['academic_style']['connective_count']}")
    print(f"    First person: {results['academic_style']['first_person_count']}")
    
    print(f"\n  Information: {results['information_density']['information_density_score']:.3f}")
    print(f"    Numbers: {results['information_density']['has_numbers']}")
    
    print(f"\n  Clarity: {results['clarity']['clarity_score']:.3f}")
    print(f"    Vague terms: {results['clarity']['vague_count']}")
    
    print(f"\n  Coherence: {results['coherence']['coherence_score']:.3f}")
    print(f"    Transitions: {results['coherence']['transition_count']}")
    
    
    
        """
        1. 길이 점수 (가중치 15%)

        단어 수를 세서 점수 부여
        149212 단어: 0.81.0점 (최적)
        175 단어: 1.0점 (완벽)
        범위 벗어날수록 감점 (최저 0.3점)

        2. 구조 점수 (가중치 30%) 

        키워드 존재 여부로 체크
        배경: "existing", "current", "however" 등 → 있으면 +0.171점
        방법: "we propose", "we present", "our approach" 등 → 있으면 +0.246점
        결과: "achieve", "outperform", "demonstrate" 등 → 있으면 +0.268점
        3가지 모두 있으면 보너스 +0.2점
        최대 1.0점

        3. 언어 점수 (가중치 20%)

        4가지 요소를 각각 체크 (각 0.25점씩)
        숫자 포함: "95%", "10x" 등 → +0.167점
        비교 표현: "better than", "outperform" 등 → +0.097점
        1인칭: "we", "our" → +0.227점
        쉼표 개수: 11개 근처(±3개) → +0.25점, 멀수록 감점

        4. 내용 점수 (가중치 20%)

        4가지 키워드 체크 (각 0.25점씩)
        평가/실험: "evaluation", "experiment" → +0.157점
        데이터셋: "ImageNet", "COCO", "dataset" → +0.102점
        성능: "performance", "accuracy" → +0.142점
        모델: "model", "architecture" → +0.186점

        5. 문장 점수 (가중치 15%)

        평균 단어 수/문장 계산
        1525 단어/문장: 0.81.0점 (최적)
        20 단어/문장: 1.0점 (완벽)
        너무 짧거나 길면 감점
        """