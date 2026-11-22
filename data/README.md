# Данные

Данные соревнования в репозиторий не входят. Положите их сюда так:

```text
data/
  train/
    signals/rec_*.npy        # (T, 6)  ЭМГ, вольты, 500 Гц
    targets/rec_*.npy        # (T, 20) углы пальцев, радианы
    metadata.csv             # recording_id, day_id, session_id, action, n_samples
    angle_info.json
  phase1_test/rec_*.npy      # этап 1 (синтетика)
  phase1_sample_submission.csv
  test/rec_*.npy             # этап 2 (настоящий тест)
  phase2_sample_submission.csv
```

В онлайн-среде StackMoreLayers данные уже подключены, вместо копирования достаточно ярлыка:

```bash
ln -s "$PLATFORM_DATASET_DIR" data
```
