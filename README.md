# 🚗 Premium Car Analytics Dashboard

<div align="center">

![Dashboard Preview](https://img.shields.io/badge/Dashboard-Premium%20Car%20Analytics-60A5FA?style=for-the-badge&logo=chart-line)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python)
![Dash](https://img.shields.io/badge/Dash-Plotly-FF6B6B?style=for-the-badge&logo=plotly)

**Advanced Vehicle Intelligence & Market Insights Platform**

*Real-time Market Intelligence | Data-Driven Decisions | Executive Analytics*

</div>

---

## 📊 Overview

Premium Car Analytics is a comprehensive dashboard application that provides deep insights into the Israeli car market. Built with **Dash** and **Plotly**, it offers interactive visualizations, smart filtering, and executive-grade analytics for vehicle comparison and market analysis.

### ✨ Key Features

- 🎯 **Smart Buyer's Guide** - Find the best deals with intelligent filtering
- 📈 **Market Analysis** - Comprehensive market trends and statistics
- ⚖️ **Group Comparison** - Head-to-head vehicle segment analysis
- 🔄 **Manufacturer Comparison** - Compare different car manufacturers
- 💎 **Best Deals Discovery** - AI-powered deal detection
- 📱 **Responsive Design** - Works seamlessly on all devices

---

## 🛠️ Technology Stack

<div align="center">

| Technology | Purpose |
|:---------:|:--------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Backend & Data Processing |
| ![Dash](https://img.shields.io/badge/Dash-0E4C92?style=flat-square&logo=plotly&logoColor=white) | Web Framework |
| ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) | Interactive Visualizations |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Data Manipulation |
| ![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?style=flat-square&logo=bootstrap&logoColor=white) | UI Components |

</div>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "visproject"
   ```

2. **Install dependencies**
   ```bash
   pip install dash dash-bootstrap-components plotly pandas numpy
   ```

3. **Run the application**
   ```bash
   python vis.py
   ```

4. **Access the dashboard**
   - Open your browser and navigate to `http://127.0.0.1:8050`

---

## 📁 Project Structure

```

│
├── vis.py                    # Main application file
├── assets/
│   └── style.css            # Custom styling
├── cars_dataset_cleaned.csv # Main dataset
└── README.md                # This file
```

---

## 🎨 Dashboard Features

### 🛒 Buyer's Guide
- **Multi-select filters** for countries, transmission types, fuel types, and owner count
- **Smart filtering** - Only applies filters when user makes selections
- **Top deals discovery** - Automatically finds best value vehicles
- **Interactive vehicle cards** - Click to view detailed information
- **Price & mileage range sliders**

### ⚖️ Group Comparison
- **Executive snapshot view** - Clean, professional comparison table
- **Head-to-head metrics** - Price per KM, Price Stability, Average Mileage, Average Price
- **Visual emphasis** - Medal icons highlight better values
- **Insight summary** - Quick executive summary of comparison results

### 🔄 Manufacturer Comparison
- **Price depreciation trends** - Visualize how prices change over time
- **Multi-manufacturer analysis** - Compare up to 5 manufacturers
- **KPI cards** - Key performance indicators at a glance

### 💎 Best Deals
- **Z-score analysis** - Statistically significant deals
- **Quality ratings** - Outstanding, Excellent, Very Good, Good
- **Interactive cards** - Hover effects and click-to-view details
- **Vehicle details modal** - Comprehensive information display

---

## 📊 Data Features

The dashboard analyzes:
- **Price** - Vehicle pricing trends and comparisons
- **Mileage** - Distance traveled analysis
- **Year** - Model year distribution
- **Transmission** - Manual vs Automatic
- **Fuel Type** - Gas, Diesel, Hybrid, Electric
- **Body Type** - Sedan, SUV, Hatchback, etc.
- **Country of Origin** - Manufacturing location
- **Owner Count** - Number of previous owners
- **Price Stability** - Market volatility metrics

---

## 🎯 Use Cases

- **Car Buyers** - Find the best deals and compare options
- **Market Analysts** - Understand market trends and patterns
- **Dealerships** - Analyze competitive positioning
- **Investors** - Evaluate market opportunities
- **Researchers** - Study automotive market dynamics

---

## 👥 Team Members

<div align="center">

### 🎓 Visualization Project - Year 3

| Member |
|:------:|
| 👤 **Gaya gur** | 
| 👤 **Moran shavit** | 
| 👤 **Tamar hagbi** | 
| 👤 **Matias Gernik** |

*Please update with actual team member names*

</div>

---

## 🎨 Design Philosophy

This dashboard follows **executive-grade design principles**:

- **Dark Theme** - Professional, easy on the eyes
- **High Contrast** - Clear hierarchy and readability
- **Minimal Color** - Color used only for emphasis and group identity
- **Typography First** - Font size and weight create hierarchy
- **Clean Layout** - No visual clutter, intentional spacing
- **Responsive** - Works on desktop, tablet, and mobile

---

## 📈 Key Metrics

The dashboard calculates and displays:

- **Price per KM** - Value for money indicator
- **Price Stability (σ)** - Standard deviation of prices
- **Average Mileage** - Mean distance traveled
- **Average Price** - Mean vehicle price
- **Z-Score Analysis** - Statistical significance of deals

---

## 🔧 Configuration

### Environment Variables

You can set a custom CSV path using:
```bash
export CAR_CSV_PATH="path/to/your/cars_dataset.csv"
```

### Default Settings

- **Year Range**: 2025 (current year)
- **Top Models**: 20 most common vehicles
- **Max Vehicles in Matrix**: 12 (default)

---

## 📝 License

This project is created for educational purposes as part of a visualization course.

---

## 🙏 Acknowledgments

- **Dash** - For the amazing web framework
- **Plotly** - For beautiful interactive visualizations
- **Bootstrap** - For responsive UI components
- **Pandas** - For powerful data manipulation

---

<div align="center">

**Built with ❤️ for Data Visualization**

![Made with Dash](https://img.shields.io/badge/Made%20with-Dash-0E4C92?style=for-the-badge&logo=plotly)
![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python)

*© 2025 Premium Car Analytics | Powered by Dash & Plotly*

</div>

---

## 📞 Contact & Support

For questions or issues, please contact the development team.

---

<div align="center">

⭐ **Star this project if you find it useful!** ⭐

</div>