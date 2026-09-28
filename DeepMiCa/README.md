# DeepMiCa 분할 모델

공식 저장소 `external/DeepMiCa`의 `Step2_Segmentation/_03_Train/UNet.py`를 수정 없이 불러옵니다. 고정 커밋은 `c2ce72cdf3265897827b8ab38fc9e1c7965ff812`이며 파일 SHA-256을 확인합니다. 원 저작권·라이선스는 `../external/DeepMiCa/LICENSE`를 따릅니다.

이 폴더는 **공개 분할망의 프로젝트용 학습·추론 구현**입니다. 논문의 전처리 및 분류를 포함한 전체 DeepMiCa 재현은 아닙니다. 사전학습 가중치는 제공하지 않습니다. 1채널은 공개 아키텍처를 유지하고, `--channels 2`는 첫 합성곱 입력을 바꾸는 명시적 변형입니다. 2채널은 기존 논문의 표현을 사용합니다: 첫 채널 `uint8/255 × 2 − 1`, 둘째 채널 `clip((uint8/255 − Gaussian19×19, σ=3) × 16, −1, 1)`, 경계는 `BORDER_REFLECT_101`입니다. 이 모드에서는 0–255 정수 영상과 `--scale 255`만 허용합니다. 학습 이미지는 네 변에 원본 영상의 문맥 9픽셀을 포함한 `(patch_size+18)²` 크기로 준비하고, 마스크는 중앙 `patch_size²` 영역에 대응시켜야 합니다. 실제 영상 경계의 문맥은 반사 경계로 추출하세요. 가우시안 처리 후 중앙 영역만 학습하므로 패치 경계의 인위적인 반사로 인한 학습·추론 차이를 방지합니다.

## 입력 및 학습

저장소 루트에서 실행합니다. 의존성은 PyTorch, NumPy, Pillow, OpenCV이며 기존 `.venv`를 사용합니다. 입력은 2차원 회색조 PNG/TIFF 또는 `.npy`; DICOM은 사전에 변환해야 합니다. 학습 입력은 정렬된 정사각 패치입니다. 1채널 이미지와 모든 마스크 크기는 `--patch-size`이며, 2채널 이미지는 위의 9픽셀 halo 규칙을 적용합니다. 마스크는 0/1 또는 0/255 이진값을 사용합니다. 암묵적 리사이즈는 없습니다. 영상 범위는 기본 0–255이며 16비트 영상은 `--scale 65535`, 이미 정규화된 영상은 `--scale 1`을 명시합니다.

CSV는 다음 필드를 요구합니다. 경로는 CSV 디렉터리 기준이며 절대경로도 가능합니다.

```csv
image,mask,patient_id,split
patches/p01.png,masks/p01.png,patient01,train
patches/p02.png,masks/p02.png,patient02,val
patches/p03.png,masks/p03.png,patient03,test
```

동일 환자의 split 혼합, 중복 이미지 경로, 영상과 마스크의 동일 경로를 거부합니다. 환자 식별자의 정확성은 원자료에서 보장해야 합니다. `train`과 `val`은 각각 필수이며 `test`는 학습·선택에 사용하지 않습니다. 원 영상 기준 환자 분할을 먼저 고정한 뒤 패치를 추출하세요.

```bash
.venv/bin/python DeepMiCa/pipeline.py train \
  --manifest /path/to/patch_manifest.csv \
  --output runs/deepmica/new_run --channels 1 \
  --patch-size 256 --batch-size 4 --epochs 16 --device cuda:0
```

새 출력 폴더만 허용합니다. SGD(momentum 0.99), 고정 학습률, 공개 코드의 3:1 hard-negative BCE를 사용합니다. 공개 손실이 NaN을 만드는 양성 없는 패치는 모든 음성 픽셀 BCE로 처리합니다. 이 예외, 고정 학습률, 증강 미사용, validation loss 최소값 선택은 원 학습 절차와의 차이입니다. `best.pt`, `last.pt`, 설정·manifest 해시, epoch별 손실, 완료 표식을 저장합니다. 체크포인트는 메타데이터와 state_dict만 포함하며 resume optimizer 상태는 지원하지 않습니다.

## 원본 크기 추론

```bash
.venv/bin/python DeepMiCa/pipeline.py infer \
  --checkpoint runs/deepmica/new_run/best.pt \
  --image /path/to/native_image.png \
  --output runs/deepmica/prediction_001 --stride 128 --threshold 0.5 --device cuda:0
```

추론 입력에는 GT나 박스가 없습니다. 체크포인트의 채널·스케일·패치 설정을 엄격하게 확인하고, 끝 타일까지 덮는 겹침 타일의 확률을 산술평균합니다. 작은 영상은 오른쪽·아래를 1채널에서는 0, 2채널에서는 논문과 같은 반사 경계로 확장한 후 원본 크기로 복구합니다. `probability.npy`, 이진 `mask.png`, 추론 설정을 저장합니다. CPU 누적 배열 두 개와 입력을 유지하므로 매우 큰 영상에는 충분한 RAM이 필요합니다.

## 실행 확인

```bash
.venv/bin/python DeepMiCa/smoke.py
```

CPU 합성 데이터만 사용해 1/2채널 순전파·역전파, 양성 없는 손실, checkpoint 왕복, 작은 영상·겹침 타일 경계, 환자 누출 차단 및 한 epoch 학습/validation 저장을 확인합니다. `smoke_result.json`은 이 실행 검증 기록이며 실제 성능 결과가 아닙니다. 본학습·공식 test 평가는 실행하지 않았습니다.

## GPU 데이터셋 평가

학습·추론은 `--device cuda:0` 또는 `cuda:1`로 실행합니다. 정량 평가는 다음과 같습니다. 평가 CSV는 원본 크기 영상과 같은 크기의 이진 GT 마스크를 지정하며 `test`만 포함해도 됩니다(2채널 학습용 halo 패치를 평가 입력으로 사용하지 않습니다).

```bash
.venv/bin/python DeepMiCa/evaluate.py \
  --manifest /path/to/native_eval.csv --split test \
  --checkpoint runs/deepmica/new_run/best.pt \
  --output runs/deepmica/evaluation_001 --device cuda:0 --threshold 0.5
```

양성 영상 macro Dice/IoU, 양성 픽셀 pooled recall, 음성 영상 FP 픽셀·영상 수와 TP/FP/FN/TN을 저장합니다. `per_image.jsonl`, `metrics.json`, `evaluation_complete.json`을 생성하고 임계값을 평가 데이터에서 최적화하지 않습니다. 평가 기본 장치는 `cuda:0`이며 CUDA가 없으면 CPU로 자동 전환하지 않습니다. `--device cpu`는 명시적인 CPU 확인용입니다. 현재 실제 GPU 실행 검증은 미완료이며 합성 CPU 검증만 수행했습니다.
