# \# ⚡ Energy Pulse

# \### Smart Energy Consumption Prediction System Using Machine Learning

# 

# Energy Pulse is a machine learning-based web application that estimates household appliance energy consumption using environmental measurements such as temperature, humidity, wind speed and atmospheric pressure.

# 

# \## 📌 Project Overview

# 

# The objective of this project is to demonstrate how machine learning can be used to estimate energy consumption based on environmental conditions.

# 

# The application uses a trained regression model and an interactive Streamlit dashboard to generate predictions.

# 

# \## ✨ Key Features

# 

# \- Energy consumption prediction.

# \- 25 environmental input features.

# \- Linear Regression model.

# \- Random Forest Regression model comparison during training.

# \- Interactive Streamlit dashboard.

# \- Model information and input feature analysis.

# 

# \## 🧠 Machine Learning Algorithms

# 

# \*\*1. Linear Regression\*\*

# 

# A supervised machine learning algorithm used to predict continuous numerical values.

# 

# \*\*2. Random Forest Regression\*\*

# 

# An ensemble learning algorithm that combines multiple decision trees to generate predictions.

# 

# \## 📊 Dataset

# 

# \*\*UCI Appliances Energy Prediction Dataset\*\*

# 

# \- 19,735 observations.

# \- 29 original columns.

# \- Target variable: `Appliances`.

# \- 25 selected input features.

# 

# The dataset contains residential building measurements, including temperature, humidity and appliance energy consumption.

# 

# \## 🛠️ Technologies Used

# 

# | Technology | Purpose |

# |---|---|

# | Python | Programming language |

# | Pandas | Data processing |

# | NumPy | Numerical operations |

# | Scikit-learn | Machine learning |

# | Joblib | Model saving and loading |

# | Streamlit | Web dashboard |

# | Google Colab | Model training |

# 

# \## ⚙️ Project Workflow

# 

# 1\. Dataset collection.

# 2\. Data preprocessing.

# 3\. Feature selection.

# 4\. Model training.

# 5\. Model evaluation.

# 6\. Model saving using Joblib.

# 7\. Energy prediction through Streamlit.

# 

# \## 📁 Project Structure

# 

# ```text

# Energy-Pulse/

# │

# ├── app.py

# ├── energy\_pulse\_model.pkl

# ├── requirements.txt

# ├── README.md

# └── .gitignore

# ```

# 

# \## 🚀 Installation and Setup

# 

# \### Prerequisites

# 

# \- Python 3.11

# \- Git

# \- Internet connection

# 

# \### 1. Clone the repository

# 

# ```bash

# git clone https://github.com/Saura8h01/Energy-Pulse.git

# ```

# 

# \### 2. Enter the project directory

# 

# ```bash

# cd Energy-Pulse

# ```

# 

# \### 3. Create a virtual environment

# 

# ```bash

# python -m venv .venv

# ```

# 

# \### 4. Activate the environment

# 

# Windows PowerShell:

# 

# ```powershell

# .\\.venv\\Scripts\\Activate.ps1

# ```

# 

# \### 5. Install dependencies

# 

# ```bash

# python -m pip install -r requirements.txt

# ```

# 

# \### 6. Run the application

# 

# ```bash

# python -m streamlit run app.py

# ```

# 

# Open the dashboard at:

# 

# `http://localhost:8501`

# 

# \## 📈 Model Evaluation

# 

# The project uses the following regression evaluation metrics:

# 

# \- \*\*MAE:\*\* Mean Absolute Error.

# \- \*\*RMSE:\*\* Root Mean Squared Error.

# \- \*\*R² Score:\*\* Coefficient of Determination.

# 

# Actual numerical evaluation results will be added after verification from the training results.

# 

# \## ⚠️ Project Limitations

# 

# \- The dataset represents a residential building, not a college campus.

# \- The model estimates consumption for the supplied observation.

# \- The current application does not forecast future energy demand.

# \- Prediction quality depends on the dataset and input measurements.

# 

# \## 🔮 Future Improvements

# 

# \- Real-time smart meter integration.

# \- Future energy-demand forecasting.

# \- Historical consumption analysis.

# \- Interactive energy usage reports.

# \- Online application deployment.

# 

# \## 👨‍💻 Author

# 

# \*\*Saura8h01\*\*

# 

# GitHub: https://github.com/Saura8h01

# 

# \---

# 

# ⭐ If you find this project interesting, consider starring the repository!

