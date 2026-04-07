# Hand and Finger Gesture Recognition

Real-time hand gesture detection and finger counting using computer vision. Built with Python and OpenCV as a university term project at Inha University in Tashkent.

## How It Works

1. **Background calibration** — The first 30 frames build a weighted average background model
2. **Hand segmentation** — Absolute difference + thresholding isolates the hand from the background
3. **Contour analysis** — Finds the largest contour (the hand) and computes its convex hull
4. **Finger counting** — Calculates the center of the palm, draws a circular ROI, and counts fingertip contours that cross the circle boundary

The system processes a live webcam feed and displays the detected finger count in real time.

## Tech Stack

`Python` `OpenCV` `NumPy` `scikit-learn`

## Running

```bash
pip install opencv-python numpy scikit-learn
python hg.py
```

Place your hand in the green ROI box on the right side of the frame. Wait ~30 frames for background calibration, then the system will start detecting gestures.

## Project Files

- `hg.py` — Main application with webcam capture, segmentation, and finger counting
- `hand.jpeg` — Sample hand image
- `*.pdf / *.docx` — University project report and presentation

## Author

**Sardor Allaberganov** — Inha University in Tashkent (CSE 16-2)
