# Xenopus Morphometric Pipeline

## 📖 Overview

The **Xenopus Morphometric Pipeline** is a specialized tool designed to automate the biometric analysis of *Xenopus laevis* tadpoles. Developed for research purposes, this application provides a user-friendly interface to process batches of microscope images, measure key morphological traits, and generate statistical reports.

The primary goal is to standardize and accelerate the measurement of:
- **Interocular Distance (IOD):** The distance between the eyes.
- **Total Body Length:** The estimated full length of the tadpole.

The tool calculates the IOD-to-length ratio, a critical metric in developmental biology and toxicology studies for assessing teratogenic effects.

---

## ✨ Features

- **Automated Batch Processing:** Analyze entire folders of tadpole images in one click.
- **Two Analysis Modes:**
  1. **Tadpole Morphometry:** Detects body and eyes, and computes the key biometric ratio.
  2. **Egg Counting:** (Coming soon) Quantifies fertilization rates.
- **Interactive Data Validation:** A built-in data editor allows for real-time correction of automated measurements ("human-in-the-loop").
- **Statistical Dashboard:**
  - **Outlier Detection:** Automatically flags statistically improbable measurements (Z-score > 3).
  - **Significance Testing:** Performs a Mann-Whitney U test to compare treatment groups against a control.
  - **Data Visualization:** Generates interactive boxplots with Plotly for clear visual comparisons.
- **One-Click Reporting:**
  - **Excel Export:** Saves all raw and corrected data to a `.xlsx` file.
  - **PDF Summary:** Generates a clean, professional PDF report with graphs and statistical tables.

---

## 🛠️ Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.9+ installed. Then, install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For systems without a graphical interface, `opencv-python-headless` can be used instead of `opencv-python`.*

---

## 🚀 How to Use

1. **Launch the Application:**
   - **Windows:** Double-click the `start.bat` file.
   - **Other OS / Manual:** Run the following command in your terminal:
     ```bash
     streamlit run src/app.py
     ```

2. **Configure Parameters (in the sidebar):**
   - **Analysis Mode:** Choose between "Tadpoles" or "Eggs".
   - **Input/Output Folders:**
     - **Input Folder:** Point to the directory containing your `.jpg` or `.png` images. The folder structure should ideally be `Condition/Replicate/image.jpg` (e.g., `PollutantA/Tank1/tadpole_01.jpg`).
     - **Output Folder:** Specify where to save the Excel and PDF reports.
   - **Scientific Settings:**
     - **Calibration (mm/pixel):** This is crucial. It's the conversion factor from pixels to millimeters, specific to your microscope and camera setup.
     - **Tail Factor:** An allometric multiplier to estimate the full body length from the detected (opaque) body, as the tail is transparent. Default is 2.6.

3. **Run the Analysis:**
   - Click the **"Lancer l'analyse 🚀"** button.
   - A progress bar will show the status of the image processing.

4. **Validate and Export:**
   - **Review Data:** After analysis, a table with all measurements will appear. Outliers are automatically flagged.
   - **Correct Data:** Use the integrated data editor to fix any obvious errors from the automated detection.
   - **Generate Reports:** Click the "Save Excel" and "Export PDF" buttons to get your final results.

---

## 📂 Project Structure
```
.
├── src/                    # Source code for backend logic
│   ├── eyes_detection.py   # Core image processing for tadpoles
│   ├── egg_counting.py     # Logic for egg analysis
│   ├── stats.py            # Statistical calculations
│   └── report.py           # PDF report generation
│
├── tests/                  # Unit tests
│
├── .gitignore              # Files to ignore in Git
├── README.md               # This file
├── app.py                  # Main Streamlit application file
├── requirements.txt        # List of Python dependencies
└── start.bat               # Windows startup script
```
