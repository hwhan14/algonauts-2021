# algonauts-2021

Clone coding the [development kit](https://github.com/Neural-Dynamics-of-Visual-Cognition-FUB/Algonauts2021_devkit) of The Algonauts Project 2021 Challenge

## Setting Up

Python version: 3.7 or 3.8 (decord, one of the dependency packages, cannot be installed in Python 3.9 or greater)

```zsh
python -m pip install -r requirements.txt
```

## About The Code

Modified parts from the baseline code are as follows.

- NN model: AlexNet → VGG16
- Regression: OLS → ridge

## 3D Visualizations

Interactive 3D surface plots are available for both left and right hemispheres of all 10 subjects.

- Sub01: [lh](plots/sub01_left.html), [rh](plots/sub01_right.html)
- Sub02: [lh](plots/sub02_left.html), [rh](plots/sub02_right.html)
- Sub03: [lh](plots/sub03_left.html), [rh](plots/sub03_right.html)
- Sub04: [lh](plots/sub04_left.html), [rh](plots/sub04_right.html)
- Sub05: [lh](plots/sub05_left.html), [rh](plots/sub05_right.html)
- Sub06: [lh](plots/sub06_left.html), [rh](plots/sub06_right.html)
- Sub07: [lh](plots/sub07_left.html), [rh](plots/sub07_right.html)
- Sub08: [lh](plots/sub08_left.html), [rh](plots/sub08_right.html)
- Sub09: [lh](plots/sub09_left.html), [rh](plots/sub09_right.html)
- Sub10: [lh](plots/sub10_left.html), [rh](plots/sub10_right.html)
