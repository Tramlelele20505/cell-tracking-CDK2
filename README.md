# CDK2 Activity Analysis Pipeline

A comprehensive Python-based tool for analyzing CDK2 activity in time-lapse microscopy images, based on the methodologies from Cappell et al., 2016 and the CDK2 activity calculation pipeline.

## Overview

This pipeline combines cell tracking algorithms with CDK2 activity calculation to provide comprehensive analysis of cell cycle dynamics. The tool processes time-lapse microscopy images to:

1. **Track individual cells** across multiple time frames
2. **Calculate CDK2 activity** based on fluorescence intensity ratios
3. **Analyze cell cycle dynamics** and activity patterns
4. **Generate comprehensive visualizations** and statistical reports

I'll teach you how to clone this GitHub repository step by step on your MacBook using VS Code.

## Prerequisites
First, make sure you have:
- **Git** installed on your Mac
- **VS Code** installed

To check if Git is installed, open Terminal and type:
```bash
git --version
```

If it's not installed, macOS will prompt you to install it.

## Step-by-Step Instructions

### Method 1: Using VS Code's Built-in Git Features (Easiest)

1. **Open VS Code**

2. **Open the Command Palette**
   - Press `Cmd + Shift + P` (or `View` → `Command Palette`)

3. **Clone the Repository**
   - Type `Git: Clone` and press Enter
   - Paste this URL: `https://github.com/Tramlelele20505/cell-tracking-CDK2.git`
   - Press Enter

4. **Choose a Location**
   - A dialog will appear asking where to save the repository
   - Navigate to where you want to save it (e.g., Documents, Desktop, or a projects folder)
   - Click "Select as Repository Destination"

5. **Open the Cloned Repository**
   - VS Code will ask if you want to open the cloned repository
   - Click "Open"

### Method 2: Using Terminal

1. **Open Terminal**
   - You can use Terminal app or VS Code's integrated terminal (`Ctrl + ~` or `View` → `Terminal`)

2. **Navigate to Where You Want to Clone**
   ```bash
   cd ~/Documents
   # or wherever you want to save it
   ```

3. **Clone the Repository**
   ```bash
   git clone https://github.com/Tramlelele20505/cell-tracking-CDK2.git
   ```

4. **Open in VS Code**
   ```bash
   cd cell-tracking-CDK2
   code .
   ```
   
   (The `code .` command opens the current directory in VS Code)

### After Cloning

Once cloned, you'll see the full repository structure including the `templates` folder you mentioned. You can navigate through the files in VS Code's Explorer panel on the left.

**Note:** You mentioned specifically the `/templates` folder. If you only want that specific folder, you'd still need to clone the entire repository, but you can then navigate to just that folder in VS Code.

Would you like help with anything specific after cloning, like setting up the project or understanding its structure?
