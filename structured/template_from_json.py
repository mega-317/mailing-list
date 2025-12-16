import copy

def create_template_from_json(data):
    """
    JSON 데이터에서 값(Value)을 제거하고 구조(Key)만 남긴 템플릿을 생성합니다.
    """
    if isinstance(data, dict):
        return {k: create_template_from_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        # 리스트인 경우, 첫 번째 아이템의 구조만 샘플로 남기고 나머지는 비웁니다.
        if len(data) > 0:
            return [create_template_from_json(data[0])]
        else:
            return []
    else:
        # 문자열, 숫자, 불리언 등은 'Place content here' 같은 플레이스홀더나 None으로 대체
        return None