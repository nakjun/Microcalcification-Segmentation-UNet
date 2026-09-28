# U-Net for Mammographic Microcalcification Segmentation: A Comparative Study

논문에 사용한 전처리, 학습, 내부 검증·test 평가, 외부 평가 및 계산비용 측정 코드를 모은 독립 Git 저장소입니다. 원본 연구 프로젝트의 상대 경로와 소스 내용을 유지했습니다.

## 코드 위치

| 단계 | 위치 |
|---|---|
| 원본 영상·마스크 전처리, 중복 확인, 데이터 분할 | `scripts/prepare_calcification_refinement.py`, `scripts/prepare_calcification_refinement_baselines.py`, `scripts/prepare_calcification_gate_data.py` |
| 공통 전처리·샘플링·손실·SAM 후보 생성 | `scripts/calcification_*_common.py`, `scripts/v3_ideas/`, 관련 공통 의존 모듈 |
| PatchUNet / U-Net / UTNet 학습·OOF 평가 | `scripts/comparative_study/` |
| UNeXt 학습·OOF 평가 | `scripts/improvement_unext_deterministic/` |
| DeepMiCa / YOLO-HLSAM 모델 및 실행 코드 | `DeepMiCa/`, `YOLO-HLSAM/` |
| 추가 모델 패치 생성·GPU 학습·내부 평가 | `scripts/additional_baselines/training/` |
| 내부 test 추론·평가·통계 | `scripts/comparative_held32/` |
| 외부 INbreast / CBIS-DDSM 추론·평가 | `scripts/external_current_models/` |
| 외부 평가 공통 의존성 | `scripts/external_comparative_inbreast/`, `scripts/external_v3/` |
| 추론 및 학습-step 비용 측정 | `scripts/additional_baselines/training/benchmark_*.py` |
| 설정 | `configs/` |
| 고정 외부 모델 소스 및 라이선스 | `external/`, 모델별 `vendor_*` 디렉터리 |

`C`, `L`, `H`, `LH`는 각각 Base, Base + Lesion, Base + Background, Base + Lesion + Background입니다. 파일명의 `held32`는 내부 test 코드의 기존 식별자입니다. 코드의 과거 명칭·주석·검증 규칙은 보관 시 변경하지 않았습니다.

## 실행 환경과 입력

Python 환경의 주요 패키지 버전은 `environment_versions.json`에 기록했습니다. 모델별 requirements도 보존했습니다. CUDA용 PyTorch/torchvision은 사용 GPU와 드라이버에 맞는 빌드가 필요합니다. 버전 목록은 전체 환경 lockfile이 아닙니다.

이 저장소는 **원본 실행 코드의 보관본이며, 데이터 없이 바로 실행되는 독립 파이프라인으로 재작성한 패키지는 아닙니다.** 원본 코드는 데이터 해시, 고정 분할, 초기 샘플링 풀, 이전 완료 기록 등을 검증합니다. 원래 상대 경로와 필요한 기록을 준비한 뒤 저장소 루트에서 실행해야 합니다. 검증을 우회하는 변경은 하지 않았습니다.

별도 준비 항목:

- CSAW-S 원본 영상·마스크와 전처리 manifest/native cache.
- INbreast와 CBIS-DDSM 원본 영상 및 평가 annotation.
- 원본 `runs/calcification_refinement_v5_seed42/`의 고정 folds/config/signature 및 샘플링 풀 등 실행 의존 기록.
- 공통 함수가 참조하는 `runs/`, `manifests/`, `docs/improvement_team/`의 원본 프로토콜·검증 기록. 이들은 코드-only 추출에 포함하지 않았습니다.
- 학습된 checkpoint와 SAM 사전학습 가중치. 코드 내부 source/hash 검증과 일치하는 파일이 필요합니다.
- 일부 원본 설정에 남아 있는 호스트별 데이터 경로 확인.

입력과 프로토콜을 준비한 뒤 사용할 학습 명령 예:

```bash
STUDY_BACKBONE=unet2d python scripts/comparative_study/train.py --mode train --variant H --fold 0 --gpu 0
python scripts/improvement_unext_deterministic/train.py --mode train --variant LH --fold 0 --gpu 0
```

새 입력으로 재실행하려면 immutable protocol과 source hash를 일관되게 새로 생성해야 하며, 기존 완료 표식을 단순 수정해서는 안 됩니다. 이 보관 작업에서 학습·추론은 실행하지 않았습니다.

## 보존 및 제외 범위

`SOURCE_MANIFEST.json`은 복사된 파일의 원본 상대 경로·SHA-256·포함 사유를 기록합니다. 원본과 byte 단위 동일성을 검증했습니다. 정적 import와 동적 로딩에 필요한 공통 코드, 모델별 테스트/검증 코드, 라이선스도 포함합니다. 과거 이름이 붙은 모듈은 현재 실행 코드의 의존성 때문에 포함되어 있습니다.

원본 데이터, 환자 영상, checkpoint, 예측 마스크, 실험 결과, 로그, 논문/PDF, 환경 디렉터리 및 상위 저장소의 Git 이력은 포함하지 않았습니다. 외부 모델의 Git 이력도 복사하지 않았습니다. 외부 코드의 원 저작권과 라이선스는 해당 파일을 따릅니다. 외부 코드를 포함한 전체 저장소에 임의의 통합 라이선스를 부여하지 않았습니다.

## 셸 워크플로

데이터 다운로드·경로 등록·원본 기록 로딩·전처리·학습·평가·tmux 실행 명령은 [workflows/README.md](workflows/README.md)에 정리했습니다. 기본 미리보기와 실제 실행을 구분하며, 실제 데이터 및 실행 기록은 저장소에 포함하지 않습니다.
