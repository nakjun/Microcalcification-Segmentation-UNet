# YOLO-HLSAM

This directory provides the underlying model components. See the [manuscript reproduction guide](../paper_results/README.md) for the final YOLO-HLSAM adaptation.

[논문](https://doi.org/10.1016/j.bspc.2025.108300)의 [공개 저장소](https://github.com/ccchg/yolo-hlsam)를 연결한 독립 학습·추론 구현입니다. 원본은 `../external/yolo-hlsam/yolo-hlsam/`, 고정 커밋은 `a29bb8e04796abb3af26d1cdc564183b68e13739`입니다.

**공개 코드 기반 프로젝트용 구현이며 원 논문의 전체 학습 절차 재현은 아닙니다.** 원 저장소 README는 HLSAM이라고 부르지만 분할 학습 코드에는 `MaskDecoderHQ`가 제공됩니다. 이 클래스를 `decoder.py`로 추출하고 하드코딩된 checkpoint 로드만 명시적 state_dict 전달로 변경했습니다. 원 논문의 미공개 모듈을 추정해 추가하지 않았습니다. 원 코드의 저작권 문구를 보존했으며 원본 checkout에는 LICENSE 파일이 없습니다.

## 구성

- `model.py`: 공개 YOLOv7-CBAM3 단일 클래스 검출기, SAM ViT-B/L/H + 공개 HQ decoder, 예측 박스→마스크 연결.
- `train.py`: 검출기/분할기 별도 학습. train split만 사용하며 최종 epoch를 고정해 저장합니다. GT 박스는 분할기 학습에서만 사용합니다.
- `infer.py`: 영상만 입력받아 검출 박스와 원본 크기 확률·이진 마스크 출력. 정답 마스크·정답 박스 인자가 없습니다.
- `data.py`: 명시적 환자 split, 입력 형상·경로 검사.
- `smoke.py`: 실제 검출기/decoder 역전파와 합성 실행 경로 검증. `smoke_result.json`은 성능 결과가 아닙니다.

## 데이터

공통 입력은 **고정 정규화된 uint8 L/RGB PNG/TIFF**입니다. 16비트·DICOM·float 입력을 묵시적으로 8비트로 변환하지 않으므로, 원 실험과 동일한 정규화를 적용해 준비하세요. JSON manifest는 아래 형식이며 경로는 JSON 파일의 디렉터리 기준입니다.

```json
[
  {"id":"p01_image1", "patient_id":"p01", "split":"train",
   "image":"images/p01.png", "mask":"masks/p01.png",
   "boxes":[[40,50,90,110]]},
  {"id":"p02_image1", "patient_id":"p02", "split":"val",
   "image":"images/p02.png", "mask":"masks/p02.png"}
]
```

`split`: train/val/test. 마스크는 원 영상과 크기가 같은 단일 채널이며 양수 값을 foreground로 처리합니다. `boxes`는 원 영상 픽셀의 `[x1,y1,x2,y2]`로 오른쪽/아래 끝은 exclusive입니다. 군집 단위 박스 제공을 권장합니다. 생략하면 **모든 양성 픽셀을 둘러싼 박스 하나**를 만듭니다(미세석회화별 connected-component 박스가 아님). 음성 영상은 빈 박스입니다. 환자 split 중복, 동일 이미지 중복, split 간 영상/마스크 경로 재사용, image==mask를 거부합니다. 각 박스 안에 양성 픽셀이 있어야 합니다.

## 두 단계 학습

저장소 루트에서 실행합니다. 모든 출력은 새 디렉터리만 허용합니다.

```bash
.venv/bin/python YOLO-HLSAM/train.py --stage detector \
  --manifest /path/to/manifest.json --output runs/yolo_hlsam/detector_run1 \
  --epochs 16 --detector-size 640 --device cuda:0

.venv/bin/python YOLO-HLSAM/train.py --stage segmentor \
  --manifest /path/to/manifest.json --output runs/yolo_hlsam/segmentor_run1 \
  --epochs 16 --model-type vit_b --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --device cuda:0
```

이 명령은 실행 예시이며 기존 논문의 비교 프로토콜로 확정된 설정이 아닙니다. batch size 1, AdamW, 고정 학습률, seed42, 증강 없음입니다. 검출기는 공개 anchor-assignment `ComputeLoss`를 사용하며 원 스크립트의 OTA·scheduler·증강을 재현하지 않습니다. 검출기 기본은 무작위 초기화이고 `--init-checkpoint`로 이 구현과 호환되는 단일 클래스 CBAM3 tensor state_dict를 지정할 수 있습니다.

분할기는 SAM을 동결하고 공개 decoder의 trainable 파라미터만 BCE+Dice로 학습합니다. 한 양성 영상의 박스를 하나 선택해 박스 안의 GT를 타깃으로 사용합니다. 음성은 분할기 단계에서 건너뛰고 검출기 학습에는 포함합니다. 기존 네 backbone의 두 채널 손실 ablation과 동등한 조건이라고 간주하면 안 됩니다. val/test는 학습에 쓰지 않으며 제공하지 않은 평가 성능을 생성하지 않습니다. 전 영상 축소 시 미세석회화가 소실될 수 있으므로 입력 크기·타일링 프로토콜은 본실험 전에 고정해야 합니다.

`epoch_016.pt`와 설정, manifest 해시, 학습 ID/환자, SAM 파일 해시, history, 완료 표식을 저장합니다. 학습 완료 표식은 성능 검증 통과를 뜻하지 않습니다. optimizer resume은 구현 범위에 포함하지 않습니다.

## GT 없는 추론

```bash
.venv/bin/python YOLO-HLSAM/infer.py \
  --image /path/to/native_image.png \
  --detector-checkpoint runs/yolo_hlsam/detector_run1/epoch_016.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --decoder-checkpoint runs/yolo_hlsam/segmentor_run1/epoch_016.pt \
  --model-type vit_b --output runs/yolo_hlsam/prediction_001 --device cuda:0
```

검출 입력은 letterbox640, SAM 입력은 긴 변1024 유지·패딩이며 마스크는 원 크기로 복원합니다. YOLO confidence0.25, NMS IoU0.45, 최대100박스, 이진 threshold0.5가 기본값입니다. 박스별 확률의 최댓값으로 합칩니다. 박스가 없으면 정확히 영인 마스크를 반환합니다. SAM 인코딩은 영상당 한 번만 수행합니다. `probability.npy`, `mask.png`, `prediction.json`에 예측 박스와 설정을 기록합니다.

세 체크포인트는 필수이며 없을 때 무작위 가중치로 추론하지 않습니다. tensor state_dict만 `weights_only=True`로 읽고 구조를 strict 검사합니다. 이 구현으로 저장한 decoder는 학습 당시 SAM 해시와 model type이 일치해야 합니다. 외부 순수 state_dict에는 이력이 없어 해당 해시 검증이 불가능합니다. 원 YOLO의 pickled model-object checkpoint를 자동으로 불러오지 않습니다.

공식 석회화 전용 가중치는 현재 내려받은 저장소에 없으며, 실제 데이터 본학습·성능 평가도 아직 하지 않았습니다.

## CPU 확인

```bash
MPLCONFIGDIR=/tmp/hlsam_matplotlib .venv/bin/python YOLO-HLSAM/smoke.py
```

독립 프로세스로 실행하세요. 원 YOLO 소스가 `models`, `utils` 절대 import를 사용하므로 다른 프로젝트 모듈과 충돌하면 명시적으로 중단합니다. Python 의존성은 `requirements.txt`에 기록했습니다.

학습 CLI와 checkpoint→추론 연결도 합성 데이터로 확인할 수 있습니다(임시 SAM 가중치 파일 약 350MB를 생성 후 삭제).

```bash
MPLCONFIGDIR=/tmp/hlsam_matplotlib .venv/bin/python YOLO-HLSAM/smoke_cli.py
```

`cli_smoke_result.json`에는 CPU 합성 1epoch 검출기/분할기 학습 및 image-only 추론 확인만 기록합니다. 실제 석회화 성능이나 사전학습 완료를 의미하지 않습니다.

## GPU 데이터셋 평가

학습·추론의 `--device cuda:0`은 실제 모델과 입력을 해당 GPU로 보냅니다. 두 단계 학습 후 정량 평가는 다음과 같습니다. 평가 JSON은 원본 크기 영상과 이진 GT 마스크를 지정하며, 평가용 GT는 지표 계산에만 사용됩니다.

```bash
.venv/bin/python YOLO-HLSAM/evaluate.py \
  --manifest /path/to/native_eval.json --split test \
  --detector-checkpoint runs/yolo_hlsam/detector_run1/epoch_016.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --decoder-checkpoint runs/yolo_hlsam/segmentor_run1/epoch_016.pt \
  --output runs/yolo_hlsam/evaluation_001 --device cuda:0 --threshold 0.5
```

양성 영상 macro Dice/IoU, 양성 픽셀 pooled recall, 음성 FP 픽셀·영상 수, TP/FP/FN/TN을 `metrics.json`에 저장합니다. `per_image.jsonl`과 `evaluation_complete.json`도 생성합니다. 평가 모델에는 이미지 경로만 전달하므로 manifest에 GT 박스가 있어도 프롬프트에 사용하지 않습니다. threshold는 고정하며 평가 데이터에서 조정하지 않습니다. 평가 기본 GPU는 `cuda:0`이고 미사용 가능 시 명시적으로 실패합니다. 실제 GPU 실행 검증은 아직 미완료이며 합성 CPU CLI 검증과 구별해야 합니다.
