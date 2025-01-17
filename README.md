![MadridGPS Banner](MadridGPS_Banner.png)

# MadridGPS 🗺️🚗

**MadridGPS** is a navigation system that calculates the **shortest** or **fastest** route between two locations in Madrid. It leverages **graph theory** to model the city streets and intersections, using datasets from the **Spanish Government Open Data Portal**.

This project was developed as part of a **Mathematical Engineering and AI** course at **Universidad Pontificia Comillas, ICAI**.

## 📜 Table of Contents
- [📌 Project Overview](#-project-overview)
- [🛠️ Installation](#️-installation)
- [⚙️ How It Works](#-how-it-works)
- [📂 Project Structure](#-project-structure)
- [🖥️ Technologies Used](#-technologies-used)
- [🙌 Credits](#-credits)

## 📌 Project Overview

MadridGPS allows users to:
- **Select two addresses** in Madrid as start and destination points.
- **Choose between shortest or fastest route calculations**.
- **Compute optimal routes** using Dijkstra’s algorithm on a **graph-based street network**.
- **Generate navigation instructions** with turn-by-turn guidance.
- **Visualize the path** on a graph representation of Madrid's street network.

## 🛠️ Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/iqueipopg/MadridGPS.git
```

### 2️⃣ Unzip the Data Folder
Since the dataset is large, it is stored as `data.zip`. Before running the program, unzip it:
```sh
unzip data.zip -d data
```
This will extract `cruces.csv` and `direcciones.csv` inside the `data/` folder.

### 3️⃣ Run the GPS Application
```sh
python src/gps.py
```
You can interact with the program via the command line.

## ⚙️ How It Works

### 📍 Graph Construction
- The graph **nodes** represent **street intersections**.
- The graph **edges** represent **streets connecting intersections**.
- Two graphs are created:
  - **Distance-based graph** (edge weight = Euclidean distance)
  - **Time-based graph** (edge weight = travel time based on speed limits)

### 🛣️ Route Calculation
- Uses **Dijkstra’s Algorithm** to find the shortest or fastest path.
- The program generates **step-by-step navigation instructions**.
- Displays the computed route using **NetworkX and Matplotlib**.

## 📂 Project Structure

```plaintext
├───data.zip  # Compressed dataset (contains cruces.csv and direcciones.csv)
├───data/     # Extracted data folder (after unzipping)
│   ├── cruces.csv  # Street intersections dataset
│   ├── direcciones.csv  # Address dataset
├───src/      # Source code
│   ├── grafo.py  # Graph data structure and algorithms
│   ├── callejero.py  # Handles dataset processing
│   ├── gps.py  # Main navigation program
│   ├── procesamiento_ficheros.py 
│   ├── navegador.py 
├───__pycache__/  # Python cache files (ignored)
└───README.md  # Project documentation
```

## 🖥️ Technologies Used

### 🔧 Development
- **Python** – Core programming language.
- **Typing** – Provides type hints for better code readability.
- **Sys** – Handles system-specific parameters and functions.
- **Math** – Mathematical functions for calculations.
- **Random** – Generates random values for simulations.
- **Re** – Regular expressions for text processing.
- **Heapq** – Implements priority queues (used in Dijkstra’s algorithm).

### 📊 Data Processing & Analysis
- **Pandas** – Data manipulation and analysis.
- **NumPy** – Numerical computations and matrix operations.
- **Chardet** – Detects character encoding in text files.

### 📡 Graph & Visualization
- **NetworkX** – Graph analysis and visualization.
- **Matplotlib** – Generates graphical representations.

## 🙌 Credits

This project was developed as part of the **Mathematical Engineering and AI** program at **Universidad Pontificia Comillas, ICAI**.

### 🎓 Special Thanks To:
- **Professors and mentors** for their guidance.
- **Universidad Pontificia Comillas, ICAI** for an excellent learning environment.
- **Open-source contributors** whose work made this possible.

### 👨‍💻 Developers:
- **Ignacio Queipo de Llano Pérez-Gascón**
- **Eugenio Ribón Novoa**

We extend our gratitude to all **open-source projects** that contributed to the development of **MadridGPS**. 🚀
