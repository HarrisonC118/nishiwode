# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a skills competition repository for the "海河工匠杯" (Haihe Craftsman Cup) containing computer vision and deep learning tasks. The repository has two main tracks:

- **学生组试题与答案/** (Student Group): Tasks focused on EAN/UPC barcode recognition
- **职工组试题与答案/** (Worker Group): Tasks focused on clothing size specification recognition

Each track contains 4 progressive tasks with skeleton code and corresponding answer files.

## Task Structure

### Student Group (学生组) - Barcode Recognition
- **task1.py**: Image preprocessing (80×80 resize, histogram plotting, Laplacian sharpening, morphological operations)
- **task2.py**: CNN training for barcode quantity classification (1-1000 classes, PyTorch)
- **task3.py**: Tkinter GUI for model inference with image upload and batch recognition
- **task4.py**: Training visualization (accuracy bar charts, loss curves using matplotlib)

### Worker Group (职工组) - Size Specification Recognition
- **task1.py**: Image preprocessing (96×96 resize, bilateral filtering denoising, morphological opening)
- **task2.py**: CRNN training with CTC loss for size text recognition (PyTorch)
- **task3.py**: Tkinter GUI for size specification recognition
- **task4.py**: Training visualization (similar to student group)

## Key Technologies

- **OpenCV (cv2)**: Image processing, resizing, filtering, morphological operations
- **PyTorch**: Deep learning framework for CNN/CRNN models
- **NumPy**: Numerical operations and array manipulation
- **Matplotlib**: Plotting histograms and training metrics
- **Tkinter**: GUI development for inference applications
- **PIL (Pillow)**: Image handling for GUI display

## Common Development Commands

### Running Individual Tasks
```bash
# Run any task file directly
python "学生组试题与答案/task1.py"
python "学生组试题与答案/task2.py"
python "职工组试题与答案/task1.py"
python "职工组试题与答案/task2.py"

# Run with answers
python "学生组试题与答案/task1答案.py"
python "职工组试题与答案/task1答案.py"
```

### Task Dependencies
- **task2** depends on **task1**: Task 1 must preprocess images and generate `processed/` directory with train/test splits
- **task3** depends on **task2**: Task 2 must train and save model (`.pth` file)
- **task4** depends on **task2**: Task 2 must generate `train_log.csv` during training

## Code Architecture

### Task 1 - Image Preprocessing Pipeline
1. **read_pairs()**: Parse `rec_gt.txt` to extract image paths and labels with specific suffix
2. **Image processing functions**:
   - Student: `plot_and_sharpen()` - Laplacian sharpening with histogram visualization
   - Worker: `plot_and_denoise()` - Bilateral filtering with histogram visualization
3. **process_one()**: Per-image pipeline (resize → grayscale → filter → denoise/sharpen → save)
4. **main()**: Train/test split, batch processing, generate `rec_gt_train.txt` and `rec_gt_test.txt`

### Task 2 - Model Training Pipeline
1. **Dataset class**: Custom PyTorch Dataset with `__init__`, `__len__`, `__getitem__`
   - Student: `BarcodeDataset` - Returns (image, label_index)
   - Worker: `SizeDataset` - Returns (image, character_indices, length) for CTC
2. **Model architecture**:
   - Student: `BarcodeCNN` - Conv layers → FC layers for classification
   - Worker: `SizeCRNN` - Conv layers → Bidirectional LSTM → CTC decoder
3. **Training loop**: Epochs with train/validation, metrics logging to CSV
4. **Model saving**: Best model saved to `best_barcode.pth` or `best_size.pth`

### Task 3 - GUI Inference Application
1. **load_model()**: Load trained model from `.pth` file
2. **upload_image()**: File dialog for image selection with PIL thumbnail preview
3. **recognize_single()**: Single image inference with preprocessing
4. **recognize_batch()**: Batch processing of multiple images
5. **Tkinter layout**: Image preview label, result display, control buttons

### Task 4 - Training Visualization
1. **Read CSV**: Load `train_log.csv` using pandas
2. **Plot metrics**: Bar chart for accuracy, line plot for loss
3. **Display**: Show all figures with `plt.show()`

## Important Implementation Notes

### File Paths
- All tasks use `Path(__file__).resolve().parent` to establish ROOT directory
- Input file: `rec_gt.txt` (tab-separated: image_path\tlabel)
- Output directory: `processed/` for preprocessed images
- Suffix filtering:
  - Student: `_crop_1.jpg`
  - Worker: `_crop_2.jpg`

### Image Dimensions
- Student group: 80×80 pixels
- Worker group: 96×96 pixels for preprocessing, 32×96 for model input

### Model Hyperparameters
- **Student**: BATCH_SIZE=32, EPOCHS=25, LR=5e-4, num_classes=1000
- **Worker**: BATCH_SIZE=32, EPOCHS=40, LR=2e-4, alphabet size=42 (includes 0-9, A-Z, and special chars)

### CTC Decoding (Worker Group Only)
- Use `decode_greedy()` to convert model output indices to text
- Remove duplicates and blank tokens (index 0)
- Alphabet: `"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ() -./"`

### Common TODO Patterns
The skeleton files contain TODO comments with:
- **要求** (Requirements): What needs to be implemented
- **提示** (Hints): Function names, parameters, or approaches
- Fill-in placeholders marked with `......`

## Debugging Tips

### Common Issues
1. **Missing processed directory**: Run task1 before task2
2. **Model file not found**: Run task2 training before task3 GUI
3. **CSV not found**: Ensure task2 logs metrics to `train_log.csv`
4. **Import errors**: Check for incomplete imports (task files have `import ...... as np`)
5. **Path issues**: Handle spaces in directory names with quotes in shell commands

### Device Selection
All training tasks check for CUDA availability:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Data Validation
- Student: `validate_number()` ensures barcode quantities are in range [1, 1000]
- Worker: Character encoding only accepts characters in `alphabet` string
