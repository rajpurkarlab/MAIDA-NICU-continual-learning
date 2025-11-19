# Demographics Directory

This directory should contain patient demographics CSV files for all 32 hospitals.

## Required Files

Download and place the following CSV files in this directory:

```
demographics/
├── Alberta_nicu_settings.csv
├── American_University_of_Beirut_nicu_settings.csv
├── Childrens_Hospital_Colorado_nicu_settings.csv
├── Chulalongkorn_University_nicu_settings.csv
├── Dr_Sardjito_Hospital_nicu_settings.csv
├── Essen_University_Hospital_nicu_settings.csv
├── Fundacion_Santa_Fe_de_Bogota_nicu_settings.csv
├── Indus_Hospital_nicu_settings.csv
├── International_Islamic_Medical_University_nicu_settings.csv
├── Istanbul_Training_Research_Hospital_nicu_settings.csv
├── King_Abdulaziz_Hospital_nicu_settings.csv
├── Kirikkale_Hospital_nicu_settings.csv
├── La_Paz_University_Hospital_nicu_settings.csv
├── Maharaj_Nakorn_Chiang_Mai_Hospital_nicu_settings.csv
├── Medical_University_of_South_Carolina_nicu_settings.csv
├── National_Cheng_Kung_University_Hospital_nicu_settings.csv
├── National_University_Hospital_nicu_settings.csv
├── Newark_Beth_Israel_Medical_Center_nicu_settings.csv
├── New_Somerset_Hospital_nicu_settings.csv
├── Osaka_Metropolitan_University_Hospital_nicu_settings.csv
├── Puerta_del_Mar_University_Hospital_nicu_settings.csv
├── SES_Hospital_nicu_settings.csv
├── Shiraz_University_nicu_settings.csv
├── Sichuan_People_Hospital_nicu_settings.csv
├── Sidra_Health_nicu_settings.csv
├── Tel_Aviv_Medical_Center_nicu_settings.csv
├── Tri-Service_General_Hospital_nicu_settings.csv
├── University_Hospital_Aachen_nicu_settings.csv
├── University_of_Alberta_nicu_settings.csv
├── University_of_Graz_nicu_settings.csv
├── University_of_Kragujevac_nicu_settings.csv
├── University_of_Linz_nicu_settings.csv
└── University_of_Tubingen_nicu_settings.csv
```

## File Format

Each CSV file contains patient demographics with the following required columns:

- **`file_name`**: Image filename (matches image ID)
- **`age_days`**: Patient age in days
- **`weight_grams`**: Patient weight in grams
- **`gestational_age_weeks`**: Gestational age in weeks

Example:
```csv
file_name,age_days,weight_grams,gestational_age_weeks
hospital_image_001.png,15,2500,38
hospital_image_002.png,3,1800,34
```

## Privacy Notice

These CSV files contain de-identified patient demographics. No personally identifiable information (PII) is included. All images are referenced by random IDs only.

## Download Instructions

See the main `data/README.md` file for download instructions.

After downloading, verify you have all 32 files:

```bash
# From repository root
ls data/demographics/*.csv | wc -l
# Should output: 32
```

## Usage

The demographics data is used by analysis scripts for:
- ETT width calibration based on patient age/weight
- Clinical performance analysis stratified by patient characteristics
- Ground truth distance calculations using patient-specific measurements

Most training scripts do not require demographics data, but analysis tools (in `analysis/`) may need them for accurate clinical metrics.
