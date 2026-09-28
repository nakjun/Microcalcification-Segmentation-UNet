# 현재 모델 외부 코호트 평가

새 결과 경로는 `runs/external_current_models_20260927/`입니다. 기존 결과는 읽기 전용으로 보존합니다.

- 조건 7개: `proposed_LH`, `unext_LH`, `unet2d_LH`, `utnet_LH`, `unet2d_H`, `deepmica`, `yolo`.
- 각 seed42, epoch16, fold0/1/2를 독립 적용하며 ensemble·외부 선택·재학습을 하지 않습니다.
- INbreast410, CBIS-DDSM 공식 Calc-Test284 전체 영상. 현재 내부 평가와 같은 native256/stride192/평균 확률/p≥.5입니다.
- 원4 모델은 BF16/batch16, 추가2는 기존 내부 평가 FP32/batch1 경로입니다. YOLO는 각 타일의 예측 박스만 사용합니다.
- 기존 `external_v3`의 gate/.9/detector union은 사용하지 않습니다. 결과는 **ungated_p5**로 구분합니다.
- image-only inventory와 기존 `native_dicom` 정규화를 사용합니다. pixel padding 제외 후 slope/intercept 및 p0.5–99.5 정규화, MONOCHROME1 반전, 원본 uint8 크기를 유지합니다.
- 고정 임계값의 packed mask를 저장합니다. threshold sweep이나 ensemble은 이 결과 파일로 지원하지 않습니다.

CPU 확인:

```bash
.venv/bin/python scripts/external_current_models/test_protocol.py
.venv/bin/python scripts/external_v3/test_evaluation.py
.venv/bin/python scripts/external_comparative_inbreast/test_inference.py
```

구현 파일을 모두 확정한 후 protocol을 동결합니다. 모델 체크포인트와 완료 근거를 확인하고 이후 소스 변경을 거부합니다.

```bash
.venv/bin/python scripts/external_current_models/infer.py --prepare
.venv/bin/python scripts/external_current_models/infer.py --dataset inbreast --condition deepmica --fold 0 --device cuda:1 --smoke
```

smoke는 첫 외부 영상 한 장을 사용하지만 `smoke/`에 별도 저장되며 정식 평가에 진입하지 못합니다. 7개 조건별 같은 명령을 적용할 수 있습니다.

전체 직렬 GPU 큐와 재개:

```bash
.venv/bin/python scripts/external_current_models/run_queue.py --device cuda:1
```

조건·fold·dataset별 분리도 지원합니다. 동일 device 큐 lock과 각 cohort lock을 적용합니다. 중단 후 같은 명령을 재실행하면 해시와 완료 표식을 확인한 영상은 재사용합니다. `queue/*.json`은 현재 단계와 PID를 기록하며 실패 시 `*.failure.txt`를 남깁니다. 큐 전체 완료는 선택한 범위 완료이고, dataset 전체 완료는 7조건×3fold 평가 완료 후에만 작성합니다.

단독 inference/evaluation:

```bash
.venv/bin/python scripts/external_current_models/infer.py --dataset cbis_test --condition unet2d_H --fold 1 --device cuda:1
.venv/bin/python scripts/external_current_models/evaluate.py --dataset cbis_test --condition unet2d_H --fold 1
```

평가에서는 선택된 checkpoint/cohort 전체 prediction과 hash 검증을 먼저 완료한 뒤 기존 annotation audit JSON을 엽니다. INbreast4339 points/2801 valid contours/67 missing XML, CBIS324 valid ROIs/2 excluded series를 고정 확인합니다. INbreast exact/radius1 point hit 및 partial-contour Dice/IoU, CBIS per-ROI overlap/coverage는 기존 평가 함수를 재사용합니다. missing annotation은 음성으로 취급하지 않으며 임상 sensitivity/specificity/FROC로 해석하지 않습니다. 각 fold 원본값과 3fold mean/sample SD를 보존합니다.

native cache 예상 약8.21GiB, 모든 packed mask를 압축하지 않았을 때 cache 포함 상한 약29.74GiB입니다. 시작 시 이 상한과8GiB 여유를 확인하고 각 영상 저장 전에도8GiB 여유를 유지합니다. shared cache는 image-only이며 DICOM/source/cache 해시를 검증합니다. 모든 체크포인트 추론 21×694=14,574회이므로 YOLO를 포함한 전체 실행은 장시간 작업입니다.
