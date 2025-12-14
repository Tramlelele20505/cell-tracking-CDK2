# CDK2 Activity Single-Cell Analysis Pipeline

A comprehensive, Python-based tool for the automated, single-cell resolution analysis of CDK2 activity dynamics in time-lapse fluorescence microscopy images. The project integrates high-accuracy deep learning segmentation (Cellpose) and advanced Bayesian inference tracking (btrack) to quantify cell cycle progression and investigate the **proliferation-quiescence decision** at a single-cell level.

##  Scientific Context and Motivation

  * **Core Research Question:** How do mammalian cells decide between continuous proliferation (cell cycle entry) and entering a reversible quiescent (resting) state G0?
  * **Biological Mechanism:** This critical cell fate decision is regulated by the dynamics of **CDK2 (Cyclin-Dependent Kinase 2) activity**. The decision point, a **bifurcation in CDK2 activity**, occurs at mitotic exit (G1 phase).
  * **Measuring Technique:** CDK2 activity is quantified using the Cytoplasm-to-Nucleus (**C/N ratio**) of the fluorescent DHB-Ven biosensor.
      * **High CDK2 Activity:** The biosensor is phosphorylated, leading to its export from the nucleus (high C/N ratio).
      * **Low CDK2 Activity:** The biosensor accumulates in the nucleus (low C/N ratio).

##  Analysis Pipeline and Workflow

The pipeline is structured into four sequential stages to transform raw time-lapse image stacks into quantitative data on single-cell CDK2 dynamics.

### Stage 1: High-Accuracy Cell Segmentation

This stage is critical for accurately defining the regions of interest (ROI) for subsequent intensity measurements.

| Component | Tool / Model | Task | Contributor |
| :--- | :--- | :--- | :--- |
| **Nucleus Segmentation** | **Cellpose** | Identify and segment nucleus boundaries in the `ch00` channel. | Vo Le Hai Dang |
| **Cytoplasm Segmentation** | **Cellpose** | Delineate the cell body (cytoplasm) from the cell mask and the nucleus mask in the `ch002` channel. | Truong Anh Phan |

### Stage 2: Quantifying Signal and Calculating CDK2 Activity

The segmented regions are used to extract fluorescent intensity values, which are then used to calculate the biological metric.

  * **Intensity Measurement:** Mean fluorescence intensity (or "whiteness") is calculated separately for the nucleus and the cytoplasm region for every cell in every frame. 
  * **CDK2 Activity Metric:** The core metric is calculated as the ratio of these intensities:

### Stage 3: Tracking and Reconstructing Cell Lineage

Individual cells (represented by the segmented masks) are linked across sequential time frames.

  * **Tool Used:** **btrack** (Bayesian Single Cell Tracking).
  * **Function:** `btrack` uses a probabilistic approach to predict cell movement and divide events, reconstructing full, deep lineage trees.
  * **Output:** The process identifies the single "best" track/lineage for analysis, which corresponds to the most complete and robust tracking path across all timepoints.

### Stage 4: Analysis, Visualization, and Output

The final processed data for the best-identified track is compiled and exported.

  * **Deliverable:** A comprehensive Excel file (`.xlsx`) containing the dynamic CDK2 activity for the selected cell lineage.

##  Technical Stack and Dependencies

The project is built around an integrated pipeline in Python, hosted by a lightweight web server for ease of use.

| Component | Technology/Library | Version/Context | Purpose |
| :--- | :--- | :--- | :--- |
| **Web Interface** | **Flask** | Python Web Framework | Handles file uploads, processes requests, and serves the results for download. |
| **Deep Learning Segmentation** | **Cellpose** | `version: 2.3.2` | High-accuracy, generalist segmentation of cell bodies and nuclei. |
| **Cell Tracking** | **btrack** | Bayesian Inference | Automated, robust tracking of cells and reconstruction of lineage trees. |
| **Image Processing** | **OpenCV (cv2)**, **Scikit-image** | Core libraries | Low-level image manipulation, filtering, and morphological operations. |
| **Data Handling** | **Pandas**, **openpyxl** | Core libraries | Tabular data manipulation and exporting the final results to Excel. |

##  Installation and Setup

### Prerequisites

1.  **Git:** Required for cloning the repository.
2.  **Python 3.x:** The project is developed and tested on Python 3.
3.  **Required Libraries:** All dependencies must be installed (see `requirements.txt`, which can be generated from the list above).

### Step 1: Clone the Repository

You can clone the project using either VS Code's built-in Git features or the terminal. The repository is hosted at: `https://github.com/Tramlelele20505/cell-tracking-CDK2.git`

#### Method A: Using VS Code's Built-in Git Features (Recommended)

1.  **Open VS Code.**
2.  **Open the Command Palette** (Press `Cmd + Shift + P` on Mac or `Ctrl + Shift + P` on Windows/Linux).
3.  Type `Git: Clone` and press Enter.
4.  Paste the URL: `https://github.com/Tramlelele20505/cell-tracking-CDK2.git` and press Enter.
5.  Choose a local directory to save the project and click "Select as Repository Destination."
6.  Click "Open" when prompted to open the cloned repository.

#### Method B: Using Terminal

1.  **Open your Terminal** application.
2.  **Navigate** to your desired project directory:
    ```bash
    cd ~/Projects
    # Replace ~/Projects with your preferred path
    ```
3.  **Clone the repository:**
    ```bash
    git clone https://github.com/Tramlelele20505/cell-tracking-CDK2.git
    ```
4.  **Navigate into the project directory:**
    ```bash
    cd cell-tracking-CDK2
    ```

### Step 2: Install Dependencies and Activate Environment

It is highly recommended to use a virtual environment to manage dependencies.

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    ```
2.  **Activate the environment:**
      * **macOS/Linux:** `source .venv/bin/activate`
      * **Windows:** `.venv\Scripts\activate`
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or manually install: pip install flask cellpose==2.3.2 btrack pandas openpyxl numpy opencv-python pillow scipy scikit-image
    ```

## Usage Guide: Running the Application

The analysis pipeline is accessible via a local web interface powered by Flask.

### 1\. Preparing Input Data

The application expects image files (e.g., `.tif`, `.png`) to be organized into two separate folders representing the two fluorescence channels:

  * **Nucleus Channel (`ch00`):** Contains images where only the nucleus is clearly visible (e.g., from a nuclear marker).
  * **Full Cell/Cytoplasm Channel (`ch002`):** Contains images where both the nucleus and cytoplasm are visible (e.g., the CDK2 biosensor signal).

The filenames for corresponding timepoints must be identical across both folders.

### 2\. Starting the Server

From the project root directory, run the main Python file:

```bash
python app.py
```

The terminal will display a message indicating the server is running, usually at: `http://127.0.0.1:5000/`

### 3\. Processing Images via Web Interface

1.  Open your web browser and navigate to the address shown by the server (e.g., `http://127.0.0.1:5000/`).
2.  **Upload Images:** Use the interface to upload the entire contents of your `ch00` folder and your `ch002` folder separately.
3.  **Start Analysis:** Click the button to initiate the full pipeline (Segmentation $\rightarrow$ Quantification $\rightarrow$ Tracking).
4.  **Download Results:** Upon completion, the application will automatically trigger the download of an Excel file named similar to `btrack_results_track[ID].xlsx`. This file contains the quantitative results for the best-identified cell track.

##  Data Output Structure

The exported Excel file (`.xlsx`) provides time-resolved data for the single best-tracked cell lineage.

| Column Name | Description | Unit / Metric |
| :--- | :--- | :--- |
| **Timepoint** | The frame number or time step in the video sequence. | Frame Index (e.g., 0, 1, 2, ...) |
| **Mean\_whiteness\_nucleus** | The average pixel intensity of the nucleus segmentation mask. | Arbitrary Fluorescence Unit (AFU) |
| **Mean\_whiteness\_cytoplasm** | The average pixel intensity of the cytoplasm segmentation mask. | Arbitrary Fluorescence Unit (AFU) |
| **CDK2\_Activity** | The calculated CDK2 activity ratio. | Dimensionless Ratio (Cytoplasm/Nucleus) |

##  Future Work

To enhance the user experience and analytical capability, the following features are planned for future development:

  * **Integrated Visualization:** Integrate a charting library (e.g., Plotly or D3.js) to generate real-time graphs of CDK2 activity trends for selected lineages directly within the browser interface.
  * **Batch Processing:** Implement functionality to process multiple time-lapse series sequentially without manual intervention.

##  Contributors

This project was developed for the Programming Integration Project under the supervision of Prof. Nguyen An Khuong.

| Name | Student ID | Assigned Tasks |
| :--- | :--- | :--- |
| **Ho Hong Phuc Nguyen** | 2352824 | Cell Tracking |
| **Hoang Thi Hang** | 2310901 | Cell Tracking, Intensity Measurement |
| **Vo Le Hai Dang** | 2352257 | Nucleus Segmentation |
| **Truong Anh Phan** | 2352881 | Cytoplasm Segmentation |
| **Le Quynh Tram** | External collaborator | Project Integration |
| **Advisor** | Prof. Nguyen An Khuong | |

Presentation Video Link: https://drive.google.com/file/d/1J7kFEp7N31p5YvhluxLGjJl-pWxmjLHH/view?usp=sharing