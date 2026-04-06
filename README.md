# AI-Enhanced Audio Plugins - Study Notes & Projects

This repository contains my personal notes, code samples, and projects developed while studying the book **"Build AI-Enhanced Audio Plugins with C++"** by Matthew Yee-King.

The goal of this repository is to document my learning progress in integrating Deep Learning (using LibTorch) with real-time audio processing.

## 📚 Reference Material
* **Book:** [Build AI-Enhanced Audio Plugins with C++](https://www.routledge.com/Build-AI-Enhanced-Audio-Plugins-with-C/Yee-King/p/book/9781032430423)
* **Author:** Matthew Yee-King
* **Focus:** C++, JUCE, LibTorch, Neural Networks for Audio.

---

## 📂 Repository Structure

The repository is organized by chapters and specific technical milestones:

* `minimal-libtorch/` – A foundational setup for linking the PyTorch C++ API (LibTorch) with a C++ project. Includes basic tensor operations and model loading.
* *(Future chapters will be added as separate directories)*

---

## 🛠️ Tech Stack
* **Language:** C++17/20
* **Frameworks:** LibTorch (PyTorch C++ API)
* **Build System:** CMake
* **Audio Tools:** JUCE

---

## 🚀 Getting Started

### Prerequisites
To build the projects in this repository, you will need:
1.  **LibTorch:** Download the C++ distribution of PyTorch from [pytorch.org](https://pytorch.org/).
2.  **CMake:** Version 3.15 or higher.
3.  **Compiler:** MSVC (Windows), Clang, or GCC.

### Building a specific project
Navigate to the desired project folder:
```bash
cd minimal-libtorch
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/your/libtorch ..
cmake --build .