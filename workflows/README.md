# 데이터 준비 및 기존 학습 프로토콜

스크립트만 제공합니다. 실제 데이터, protocol snapshot, 가중치, 결과는 Git에 포함하지 않습니다. 이 파일을 추가하는 과정에서 다운로드·학습·추론은 실행하지 않았습니다.

## 1. 데이터 확보

공식 배포 기준 확인일: 2026-09-28.

- CSAW-S: https://zenodo.org/records/4030660 — 파일 접근 제한. 공식 페이지에서 사용 신청 후 승인된 원본 파일을 확보합니다.
- INbreast: 원 배포기관에서 확보한 DICOM/XML release를 사용합니다. 검증되지 않은 공개 미러를 자동 다운로드하지 않습니다.
- CBIS-DDSM: https://www.cancerimagingarchive.net/collection/cbis-ddsm/ — 공식 페이지/NBIA 다운로드 절차를 사용합니다. 본 연구 외부 평가는 Calc-Test 전체 영상 및 해당 ROI annotation입니다. JPEG 변환본으로 대체하지 않습니다.

직접 HTTPS 다운로드 URL과 신뢰할 수 있는 SHA-256을 확보한 경우:

```bash
# URL은 인증 토큰을 포함할 수 있으므로 설정 파일이나 Git에 저장하지 않습니다.
export DATA_URL='https://provider.example/authorized-archive'
export DATA_SHA256='실제_64자리_SHA256'
bash workflows/download_data.sh csaw  # inbreast 또는 cbis도 가능
unset DATA_URL DATA_SHA256
```

`download_data.sh`는 직접 실행 시 다운로드합니다. 다운로드한 archive는 `datasets/downloads/`에 저장하며 체크섬 일치 시에만 완료 파일로 바꿉니다. 압축 해제는 배포 형식에 맞게 별도 수행합니다. 기존 파일을 덮어쓰지 않습니다. 공개 URL을 모르는 상태에서 명령만 실행하면 공식 준비 안내 후 종료합니다.

## 2. 데이터 경로 등록 및 원본 실행 기록

아래 스크립트는 기본적으로 **명령만 출력**합니다. 실제 실행은 첫 인자로 `--execute`를 넣습니다.

```bash
bash workflows/load_data.sh csaw /data/CSAW-S
bash workflows/load_data.sh inbreast /data/INbreast
bash workflows/load_data.sh cbis /data/CBIS-DDSM
# 실행 예: bash workflows/load_data.sh --execute csaw /data/CSAW-S
```

이 단계는 이미 압축 해제한 폴더를 `datasets/` 아래 심볼릭 링크로 등록합니다. 기존 Python 코드의 실제 로더는 원본 manifest의 경로를 사용합니다. 이 등록만으로 학습용 manifest/분할이 만들어지거나 절대 경로가 자동 변경되지는 않습니다.

기존 프로토콜의 실행 기록이 있는 환경에서 재사용할 경우:

```bash
bash workflows/load_data.sh artifacts /path/to/original/project
```

`--execute`를 지정하면 원본 `runs`, `reports`, `manifests`, `docs`를 **독립 복사**합니다. 변경 가능한 run 폴더를 원본에 링크하지 않습니다. 대용량 복사이므로 실제 실행 전에 출력 경로와 여유 공간을 확인합니다. 대상 폴더가 존재하면 병합/덮어쓰기 없이 중단합니다. 기록에 저장된 절대 경로 및 source hash는 그대로 유지하며 원래 검증을 통과해야 합니다. 이 작업은 새 데이터에서 완전히 새 프로토콜을 생성하는 기능이 아닙니다.

## 3. 전처리 / 로딩

```bash
bash workflows/preprocess.sh native      # 원본 manifest 기반 영상·마스크 감사/캐시/분할
bash workflows/preprocess.sh baseline    # 기존 baseline 후보/마스크 의존 자료
bash workflows/preprocess.sh additional  # 추가 모델 고정 패치 및 manifest
bash workflows/preprocess.sh test        # 내부 test 프로토콜
bash workflows/preprocess.sh external    # 기존 외부 cohort manifest를 고정 protocol에 연결
```

각 단계는 기존 Python 진입점을 사용합니다. 단계별 선행 입력이 필요하며, 원본 데이터 다운로드만으로 모든 historical dependency가 충족되지는 않습니다. 기존 분할 124 development / 32 test와 seed42를 재선정하지 않습니다. 외부 annotation audit JSON 생성은 이 wrapper의 기능이 아니며 원본 입력 기록이 필요합니다.

## 4. 네 backbone 학습

```bash
export GPU_INDEX=0
# model: proposed(PatchUNet), unet2d, utnet, unext
bash workflows/train.sh unet2d prepare
bash workflows/train.sh unet2d smoke
bash workflows/train.sh unet2d warmup
bash workflows/train.sh unet2d train
bash workflows/train.sh unet2d infer
bash workflows/train.sh unet2d evaluate
```

- seed42, fold0–2, C/L/H/LH 네 조건. C=Base, L=Base + Lesion, H=Base + Background, LH=Base + Lesion + Background.
- 16 epochs, epoch당 2,048 patches, batch16, 원본 deterministic/BF16 설정 유지.
- warmup은 모든 12개 조합에서 3 epochs를 수행하고 `verify_warmup.py`로 동일 초기 상태를 확인합니다. 이후 train은 원본 구현에 따라 epoch16까지 진행합니다.
- `preflight.json` 및 warm-up 검증 표식이 필요합니다. smoke 실행만으로 독립 사전검토가 승인된 것으로 취급하지 않으며, wrapper는 승인 표식을 만들거나 hash 검사를 우회하지 않습니다.
- evaluate의 출력에는 이전 정책도 포함되므로, 논문의 OOF 수치는 `ungated_p5`를 사용합니다. 내부 test/외부 결과와 섞지 않습니다.

## 5. 추가 모델

```bash
bash workflows/train.sh deepmica train
bash workflows/train.sh yolo train
```

원본 worker를 그대로 호출합니다. GPU는 **cuda:1**로 고정되어 있으며 GPU_INDEX로 바뀌지 않습니다. `.venv/bin/python`, protocol, 준비된 patch, 모델별 GPU preflight 기록이 필요합니다. DeepMiCa는 16 epochs, batch16, SGD lr0.01; YOLO-HLSAM은 detector/segmentor 각16 epochs, seed42입니다. SAM 가중치는 `runs/mammo_hypothesis/weights/sam_vit_b.pth`가 필요합니다. 학습 후 worker가 내부 추론·평가를 수행합니다. 원본 worker는 기존 출력 디렉터리가 있으면 실패하므로 중복 실행하거나 결과를 지워 재시작하지 않습니다.

## 6. 평가 및 tmux 실행

```bash
bash workflows/evaluate.sh test
bash workflows/evaluate.sh deepmica
bash workflows/evaluate.sh yolo
DEVICE=cuda:1 bash workflows/evaluate.sh external

# 실제 장시간 작업은 tmux에서 실행; 이 명령은 미리보기가 아닌 실제 실행입니다.
bash workflows/tmux_run.sh unet_training train unet2d train
bash workflows/tmux_run.sh external_validation evaluate external
```

GPU 학습을 수행하기 전에 prepare/smoke/사전검토/warmup 단계를 완료해야 합니다. 모든 wrapper는 실패 시 중단합니다. 원본 데이터 및 학습 프로토콜 검사 조건을 유지하며 임의 checkpoint/threshold 선택을 하지 않습니다.
