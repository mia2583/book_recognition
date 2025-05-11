import os
import yaml
import numpy as np

def save_calibration_to_yaml(filename, K, D, image_width=None, image_height=None):
    """카메라 매트릭스(K)와 왜곡 계수(D)를 YAML 파일에 저장합니다."""
    data = {
        "K": K.tolist(),
        "D": D.tolist()
    }
    if image_width is not None and image_height is not None:
        data["image_width"] = image_width
        data["image_height"] = image_height

    try:
        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)  # 가독성 좋게 저장
        print(f"[INFO] 캘리브레이션 데이터 저장 완료: {filename}")
    except Exception as e:
        print(f"[ERROR] YAML 파일 저장 중 오류 발생 ({filename}): {e}")

def load_calibration_from_yaml(filename):
    """YAML 파일에서 카메라 매트릭스(K)와 왜곡 계수(D)를 로드합니다."""
    if not os.path.exists(filename):
        print(f"[WARN] 캘리브레이션 파일을 찾을 수 없습니다: {filename}")
        return None, None, None, None
    try:
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        K = np.array(data["K"], dtype=np.float32)
        D = np.array(data["D"], dtype=np.float32)
        image_width = data.get("image_width")
        image_height = data.get("image_height")
        return K, D, image_width, image_height

    except FileNotFoundError:
        print(f"[ERROR] 캘리브레이션 파일을 찾을 수 없습니다: {filename}")
        return None, None, None, None

    except KeyError as e:
        print(f"[ERROR] YAML 파일 형식 오류: 필수 키 '{e}'가 없습니다 ({filename}).")
        return None, None, None, None

    except yaml.YAMLError as e:
        print(f"[ERROR] YAML 파싱 오류 ({filename}): {e}")
        return None, None, None, None

    except Exception as e:
        print(f"[ERROR] 캘리브레이션 파일 로드 중 예기치 않은 오류 발생 ({filename}): {e}")
        return None, None, None, None


